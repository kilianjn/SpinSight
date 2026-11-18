//! Background simulation thread: runs the k-space simulation and reconstruction for the most
//! recent request, keeping the GUI responsive while parameters are changed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::imaging::{SimOutput, Simulator};
use crate::phantom::{self, Phantom, PhantomKspace, Progress};
use crate::protocol::{Inputs, Protocol};

pub struct Job {
    pub id: u64,
    pub inputs: Inputs,
    pub protocol: Arc<Protocol>,
    pub phantom: Arc<Phantom>,
    pub renew_noise: bool,
}

pub struct JobResult {
    pub id: u64,
    pub output: Arc<SimOutput>,
    pub protocol: Arc<Protocol>,
    pub inputs: Inputs,
}

/// Shared status of the worker (for progress display)
#[derive(Default)]
pub struct Status {
    /// name of the phantom whose k-space is being prepared
    pub preparing_phantom: Mutex<Option<String>>,
    pub progress: Arc<Progress>,
    pub busy: AtomicBool,
    pub last_phantom_secs: Mutex<Option<(String, f64, bool)>>,
}

pub struct Worker {
    tx: Sender<Job>,
    rx: Receiver<JobResult>,
    pub status: Arc<Status>,
}

impl Worker {
    /// `notify` is called (from the worker thread) whenever there is something new to show
    pub fn spawn(notify: impl Fn() + Send + 'static) -> Worker {
        let (tx, job_rx) = channel::<Job>();
        let (res_tx, rx) = channel::<JobResult>();
        let status = Arc::new(Status::default());
        let st = status.clone();
        std::thread::Builder::new()
            .name("simulation".into())
            .spawn(move || {
                let mut sim = Simulator::default();
                let mut kspaces: HashMap<u64, Arc<PhantomKspace>> = HashMap::new();
                while let Ok(mut job) = job_rx.recv() {
                    // only the newest request matters
                    let mut renew = job.renew_noise;
                    loop {
                        match job_rx.try_recv() {
                            Ok(j) => {
                                renew |= j.renew_noise;
                                job = j;
                            }
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => return,
                        }
                    }
                    st.busy.store(true, Ordering::Relaxed);
                    let pk = match kspaces.get(&job.phantom.hash) {
                        Some(k) => k.clone(),
                        None => {
                            let cached = phantom::is_cached(&job.phantom);
                            *st.preparing_phantom.lock().unwrap() = Some(job.phantom.name.clone());
                            notify();
                            let t = Instant::now();
                            let k = Arc::new(phantom::load_kspace(&job.phantom, Some(&st.progress)));
                            *st.last_phantom_secs.lock().unwrap() = Some((job.phantom.name.clone(), t.elapsed().as_secs_f64(), cached));
                            *st.preparing_phantom.lock().unwrap() = None;
                            kspaces.insert(job.phantom.hash, k.clone());
                            k
                        }
                    };
                    if renew {
                        sim.renew_noise();
                    }
                    let output = Arc::new(sim.run(&job.protocol, &job.inputs, &job.phantom, &pk));
                    st.busy.store(false, Ordering::Relaxed);
                    if res_tx.send(JobResult { id: job.id, output, protocol: job.protocol, inputs: job.inputs }).is_err() {
                        return;
                    }
                    notify();
                }
            })
            .expect("could not start the simulation thread");
        Worker { tx, rx, status }
    }

    pub fn submit(&self, job: Job) {
        let _ = self.tx.send(job);
    }

    /// The newest finished result, if any
    pub fn poll(&self) -> Option<JobResult> {
        let mut last = None;
        while let Ok(r) = self.rx.try_recv() {
            last = Some(r);
        }
        last
    }
}

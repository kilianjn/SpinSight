//! Voronoi density compensation weights for non-Cartesian sampling
//! (port of `mrinufft.density.voronoi`, including its edge-cell extrapolation).

use std::collections::HashMap;

use delaunator::{next_halfedge, triangulate, Point, EMPTY};

/// Voronoi cell areas of `pts`; `f64::INFINITY` for open (hull) cells.
/// Exact duplicates share the cell of their first occurrence.
fn cell_areas(pts: &[[f64; 2]]) -> Vec<f64> {
    // deduplicate exactly equal points (Qhull assigns duplicates to the same region)
    let mut unique: Vec<[f64; 2]> = Vec::new();
    let mut map = Vec::with_capacity(pts.len());
    let mut seen: HashMap<(u64, u64), usize> = HashMap::new();
    for p in pts {
        let key = (p[0].to_bits(), p[1].to_bits());
        let id = *seen.entry(key).or_insert_with(|| {
            unique.push(*p);
            unique.len() - 1
        });
        map.push(id);
    }
    let points: Vec<Point> = unique.iter().map(|p| Point { x: p[0], y: p[1] }).collect();
    let tri = triangulate(&points);
    let n = points.len();
    let mut area = vec![f64::INFINITY; n];
    if tri.triangles.is_empty() {
        return map.iter().map(|&i| area[i]).collect();
    }
    let ntri = tri.triangles.len() / 3;
    let circumcenters: Vec<[f64; 2]> = (0..ntri)
        .map(|t| {
            let a = &points[tri.triangles[3 * t]];
            let b = &points[tri.triangles[3 * t + 1]];
            let c = &points[tri.triangles[3 * t + 2]];
            circumcenter(a, b, c)
        })
        .collect();
    // one incoming half-edge per point
    let mut incoming = vec![EMPTY; n];
    for e in 0..tri.triangles.len() {
        let endpoint = tri.triangles[next_halfedge(e)];
        if incoming[endpoint] == EMPTY || tri.halfedges[e] == EMPTY {
            incoming[endpoint] = e;
        }
    }
    let mut poly: Vec<[f64; 2]> = Vec::new();
    for p in 0..n {
        let start = incoming[p];
        if start == EMPTY {
            continue;
        }
        poly.clear();
        let mut e = start;
        let mut closed = false;
        loop {
            poly.push(circumcenters[e / 3]);
            let outgoing = next_halfedge(e);
            let next = tri.halfedges[outgoing];
            if next == EMPTY {
                break;
            }
            if next == start {
                closed = true;
                break;
            }
            e = next;
            if poly.len() > 10_000 {
                break;
            }
        }
        if closed {
            area[p] = polygon_area(&poly);
        }
    }
    map.iter().map(|&i| area[i]).collect()
}

fn circumcenter(a: &Point, b: &Point, c: &Point) -> [f64; 2] {
    let (bx, by) = (b.x - a.x, b.y - a.y);
    let (cx, cy) = (c.x - a.x, c.y - a.y);
    let d = 2.0 * (bx * cy - by * cx);
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (cy * b2 - by * c2) / d;
    let uy = (bx * c2 - cx * b2) / d;
    [a.x + ux, a.y + uy]
}

fn polygon_area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        s += p[i][0] * p[j][1] - p[j][0] * p[i][1];
    }
    s.abs() / 2.0
}

/// numpy `percentile(a, q)` with linear interpolation (including its NaN behaviour with infinities)
fn percentile(a: &[f64], q: f64) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let virt = q / 100.0 * (n - 1) as f64;
    let lo = virt.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let t = virt - lo as f64;
    let (a, b) = (v[lo], v[hi]);
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// Least-squares cubic fit like `np.polynomial.Polynomial.fit(x, y, 3)`; returns an evaluator
fn polyfit3(x: &[f64], y: &[f64]) -> impl Fn(f64) -> f64 {
    let (xmin, xmax) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let (off, scl) = if xmax > xmin {
        let scl = 2.0 / (xmax - xmin);
        (-1.0 - xmin * scl, scl)
    } else {
        (0.0, 1.0)
    };
    // normal equations on the mapped variable
    let mut ata = vec![vec![0.0; 4]; 4];
    let mut aty = vec![0.0; 4];
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let u = off + scl * xi;
        let pw = [1.0, u, u * u, u * u * u];
        for r in 0..4 {
            aty[r] += pw[r] * yi;
            for c in 0..4 {
                ata[r][c] += pw[r] * pw[c];
            }
        }
    }
    let coef = if x.len() >= 4 { crate::sequence::solve_dense(ata, aty) } else { vec![f64::NAN; 4] };
    move |xv: f64| {
        let u = off + scl * xv;
        coef[0] + u * (coef[1] + u * (coef[2] + u * coef[3]))
    }
}

/// Density compensation weights (`mrinufft.density.voronoi`, returning `1 / _normalize_weights(wi)`)
pub fn voronoi_density(traj: &[[f64; 2]]) -> Vec<f64> {
    // flat_traj(normalize="unit")
    let max_abs = traj.iter().fold(0.0f64, |m, p| m.max(p[0].abs()).max(p[1].abs()));
    let mut pts: Vec<[f64; 2]> = traj.to_vec();
    if max_abs - 1e-4 > 0.5 {
        for p in pts.iter_mut() {
            p[0] /= 2.0 * std::f64::consts::PI;
            p[1] /= 2.0 * std::f64::consts::PI;
        }
    }
    if max_abs >= 0.5 {
        for p in pts.iter_mut() {
            p[0] = (p[0] + 0.5).rem_euclid(1.0) - 0.5;
            p[1] = (p[1] + 0.5).rem_euclid(1.0) - 0.5;
        }
    }
    // _voronoi_unique (applied to all points, as in the reference implementation)
    let mut wi = cell_areas(&pts);
    let thresh = percentile(&wi, 95.0);
    for w in wi.iter_mut() {
        if *w > thresh {
            *w = f64::INFINITY;
        }
    }
    let rho: Vec<f64> = pts.iter().map(|p| p[0] * p[0] + p[1] * p[1]).collect();
    let rho_max = rho.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (gx, gy): (Vec<f64>, Vec<f64>) = rho.iter().zip(wi.iter()).filter(|(r, w)| **r > 0.6 * rho_max && w.is_finite()).map(|(r, w)| (*r, *w)).unzip();
    let poly = polyfit3(&gx, &gy);
    for (w, r) in wi.iter_mut().zip(rho.iter()) {
        if w.is_infinite() {
            *w = poly(*r);
        }
    }
    // split the weight of the k-space centre evenly among its duplicates
    let zeros: Vec<usize> = (0..pts.len()).filter(|&i| pts[i][0].abs() + pts[i][1].abs() == 0.0).collect();
    if !zeros.is_empty() {
        let n0 = zeros.len() as f64;
        for &i in &zeros {
            wi[i] /= n0;
        }
    }
    // 1 / _normalize_weights(wi) = wi * Σ(1/wi)
    let inv_sum: f64 = wi.iter().map(|w| 1.0 / w).sum();
    wi.iter().map(|w| w * inv_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_cells_have_unit_area() {
        let mut pts = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                pts.push([i as f64, j as f64]);
            }
        }
        let a = cell_areas(&pts);
        assert!((a[5 * 10 + 5] - 1.0).abs() < 1e-9);
        assert!(a[0].is_infinite());
    }
}

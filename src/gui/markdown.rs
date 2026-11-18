//! Tiny markdown renderer for the help tooltips (headings, paragraphs, *italic*, **bold**).

use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, FontId, Ui};

fn inline(job: &mut LayoutJob, text: &str, size: f32, color: Color32, strong_color: Color32, base_bold: bool) {
    let mut bold = base_bold;
    let mut italic = false;
    let mut buf = String::new();
    let flush = |job: &mut LayoutJob, buf: &mut String, bold: bool, italic: bool| {
        if buf.is_empty() {
            return;
        }
        let mut f = TextFormat { font_id: FontId::proportional(size), color: if bold { strong_color } else { color }, italics: italic, ..Default::default() };
        if bold {
            // emulate bold with a slightly stronger colour and size
            f.font_id = FontId::proportional(size + 0.5);
        }
        job.append(buf, 0.0, f);
        buf.clear();
    };
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            flush(job, &mut buf, bold, italic);
            bold = !bold;
            i += 2;
        } else if chars[i] == '*' {
            flush(job, &mut buf, bold, italic);
            italic = !italic;
            i += 1;
        } else {
            buf.push(chars[i]);
            i += 1;
        }
    }
    flush(job, &mut buf, bold, italic);
}

/// Render `md` into the ui with the given maximum width
pub fn show(ui: &mut Ui, md: &str, width: f32) {
    let visuals = ui.visuals().clone();
    let color = visuals.text_color();
    let strong = visuals.strong_text_color();
    ui.set_max_width(width);
    let mut first = true;
    for para in md.split("\n\n") {
        for line in para.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let mut job = LayoutJob::default();
            job.wrap.max_width = width;
            if let Some(h) = line.strip_prefix("## ") {
                inline(&mut job, h, 15.0, strong, strong, true);
            } else if let Some(h) = line.strip_prefix("# ") {
                inline(&mut job, h, 18.0, strong, strong, true);
            } else {
                inline(&mut job, line, 13.5, color, strong, false);
            }
            ui.label(job);
        }
        if !first {
            ui.add_space(2.0);
        }
        first = false;
        ui.add_space(4.0);
    }
}

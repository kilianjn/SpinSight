//! Minimal SVG reader for phantom files: extracts `<path>` elements, their fill colour and
//! their closed polygon subpaths (mirrors the parts of `svgpathtools` used by SpinSight).

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegKind {
    Line,
    /// Bézier curves and arcs: not supported as phantom geometry
    Curve,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub kind: SegKind,
    /// (x, y)
    pub start: [f64; 2],
    pub end: [f64; 2],
}

#[derive(Debug, Clone)]
pub struct SvgPath {
    pub fill: Option<String>,
    pub segments: Vec<Segment>,
    /// affine transform [a, b, c, d, e, f]: x' = a x + c y + e, y' = b x + d y + f
    pub transform: [f64; 6],
}

fn parse_style(s: &str) -> Vec<(String, String)> {
    s.split(';').filter_map(|a| a.split_once(':')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect()
}

fn get_styles(doc: &roxmltree::Document) -> Vec<(String, Vec<(String, String)>)> {
    let mut styles = Vec::new();
    for node in doc.root_element().children().filter(|n| n.has_tag_name("style")) {
        let text: String = node.descendants().filter_map(|n| n.text()).collect();
        let mut rest = text.as_str();
        while let Some(dot) = rest.find('.') {
            rest = &rest[dot + 1..];
            let name_end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(rest.len());
            let name = rest[..name_end].to_string();
            let Some(open) = rest.find('{') else { break };
            let Some(close) = rest.find('}') else { break };
            if open < close && rest[name_end..open].trim().is_empty() {
                styles.push((name, parse_style(&rest[open + 1..close])));
            }
            rest = &rest[close.min(rest.len() - 1) + 1..];
        }
    }
    styles
}

fn tokenize(d: &str) -> Vec<Tok> {
    let mut toks = Vec::new();
    let b = d.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() && c != 'e' && c != 'E' {
            toks.push(Tok::Cmd(c));
            i += 1;
        } else if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() {
            let start = i;
            i += 1;
            let mut seen_dot = c == '.';
            let mut seen_exp = false;
            while i < b.len() {
                let ch = b[i] as char;
                if ch.is_ascii_digit() {
                    i += 1;
                } else if ch == '.' && !seen_dot && !seen_exp {
                    seen_dot = true;
                    i += 1;
                } else if (ch == 'e' || ch == 'E') && !seen_exp {
                    seen_exp = true;
                    i += 1;
                    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
                        i += 1;
                    }
                } else {
                    break;
                }
            }
            if let Ok(v) = d[start..i].parse::<f64>() {
                toks.push(Tok::Num(v));
            }
        } else {
            i += 1; // separators
        }
    }
    toks
}

#[derive(Debug, Clone, Copy)]
enum Tok {
    Cmd(char),
    Num(f64),
}

/// Parse SVG path data into line/curve segments (like `svgpathtools.parse_path`)
pub fn parse_path(d: &str) -> Vec<Segment> {
    let toks = tokenize(d);
    let mut segs = Vec::new();
    let mut i = 0;
    let mut cmd = ' ';
    let mut cur = [0.0, 0.0];
    let mut start = [0.0, 0.0];
    let mut first = true;
    let num = |i: &mut usize| -> Option<f64> {
        if let Some(Tok::Num(v)) = toks.get(*i) {
            *i += 1;
            Some(*v)
        } else {
            None
        }
    };
    while i < toks.len() {
        if let Tok::Cmd(c) = toks[i] {
            cmd = c;
            i += 1;
        } else if cmd == ' ' {
            i += 1;
            continue;
        }
        let rel = cmd.is_ascii_lowercase();
        let base = |cur: [f64; 2]| if rel { cur } else { [0.0, 0.0] };
        match cmd.to_ascii_uppercase() {
            'M' => {
                let (Some(x), Some(y)) = (num(&mut i), num(&mut i)) else { break };
                let b = if first { [0.0, 0.0] } else { base(cur) };
                cur = [b[0] + x, b[1] + y];
                start = cur;
                first = false;
                // subsequent pairs are implicit line-tos
                cmd = if rel { 'l' } else { 'L' };
            }
            'L' => {
                let (Some(x), Some(y)) = (num(&mut i), num(&mut i)) else { break };
                let b = base(cur);
                let p = [b[0] + x, b[1] + y];
                segs.push(Segment { kind: SegKind::Line, start: cur, end: p });
                cur = p;
            }
            'H' => {
                let Some(x) = num(&mut i) else { break };
                let p = [if rel { cur[0] + x } else { x }, cur[1]];
                segs.push(Segment { kind: SegKind::Line, start: cur, end: p });
                cur = p;
            }
            'V' => {
                let Some(y) = num(&mut i) else { break };
                let p = [cur[0], if rel { cur[1] + y } else { y }];
                segs.push(Segment { kind: SegKind::Line, start: cur, end: p });
                cur = p;
            }
            'Z' => {
                if cur != start {
                    segs.push(Segment { kind: SegKind::Line, start: cur, end: start });
                }
                cur = start;
                cmd = ' ';
            }
            other => {
                // curves / arcs: consume their parameters and keep track of the end point
                let n = match other {
                    'C' => 6,
                    'S' | 'Q' => 4,
                    'T' => 2,
                    'A' => 7,
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let mut vals = Vec::with_capacity(n);
                for _ in 0..n {
                    match num(&mut i) {
                        Some(v) => vals.push(v),
                        None => break,
                    }
                }
                if vals.len() < n {
                    break;
                }
                let b = base(cur);
                let p = [b[0] + vals[n - 2], b[1] + vals[n - 1]];
                segs.push(Segment { kind: SegKind::Curve, start: cur, end: p });
                cur = p;
            }
        }
    }
    segs
}

fn parse_transform(s: &str) -> [f64; 6] {
    let mut m = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut rest = s;
    while let Some(open) = rest.find('(') {
        let name = rest[..open].trim().trim_start_matches(',').trim();
        let Some(close) = rest.find(')') else { break };
        let args: Vec<f64> = rest[open + 1..close].split(|c: char| c == ',' || c.is_whitespace()).filter(|t| !t.is_empty()).filter_map(|t| t.parse().ok()).collect();
        let t = match (name, args.as_slice()) {
            ("matrix", [a, b, c, d, e, f]) => [*a, *b, *c, *d, *e, *f],
            ("translate", [x]) => [1.0, 0.0, 0.0, 1.0, *x, 0.0],
            ("translate", [x, y]) => [1.0, 0.0, 0.0, 1.0, *x, *y],
            ("scale", [s]) => [*s, 0.0, 0.0, *s, 0.0, 0.0],
            ("scale", [sx, sy]) => [*sx, 0.0, 0.0, *sy, 0.0, 0.0],
            ("rotate", [a]) => {
                let (s, c) = a.to_radians().sin_cos();
                [c, s, -s, c, 0.0, 0.0]
            }
            _ => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };
        m = compose(m, t);
        rest = &rest[close + 1..];
    }
    m
}

/// m ∘ t (apply t first)
fn compose(m: [f64; 6], t: [f64; 6]) -> [f64; 6] {
    [
        m[0] * t[0] + m[2] * t[1],
        m[1] * t[0] + m[3] * t[1],
        m[0] * t[2] + m[2] * t[3],
        m[1] * t[2] + m[3] * t[3],
        m[0] * t[4] + m[2] * t[5] + m[4],
        m[1] * t[4] + m[3] * t[5] + m[5],
    ]
}

pub fn apply_transform(m: &[f64; 6], p: [f64; 2]) -> [f64; 2] {
    [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]]
}

/// All `<path>` elements of an SVG document
pub fn read_paths(svg: &str) -> Result<Vec<SvgPath>, String> {
    let doc = roxmltree::Document::parse(svg).map_err(|e| format!("invalid SVG: {e}"))?;
    let styles = get_styles(&doc);
    let mut paths = Vec::new();
    for node in doc.descendants().filter(|n| n.has_tag_name("path")) {
        let fill = if let Some(style) = node.attribute("style") {
            parse_style(style).into_iter().find(|(k, _)| k == "fill").map(|(_, v)| v)
        } else if let Some(class) = node.attribute("class") {
            styles.iter().find(|(n, _)| n == class).and_then(|(_, st)| st.iter().find(|(k, _)| k == "fill").map(|(_, v)| v.clone()))
        } else {
            None
        };
        let segments = parse_path(node.attribute("d").unwrap_or(""));
        let transform = parse_transform(node.attribute("transform").unwrap_or(""));
        paths.push(SvgPath { fill, segments, transform });
    }
    Ok(paths)
}

/// Split segments into continuous subpaths (`Path.continuous_subpaths`)
pub fn continuous_subpaths(segs: &[Segment]) -> Vec<&[Segment]> {
    let mut out = Vec::new();
    if segs.is_empty() {
        return out;
    }
    let mut start = 0;
    for i in 0..segs.len() - 1 {
        if segs[i].end != segs[(i + 1) % segs.len()].start {
            out.push(&segs[start..=i]);
            start = i + 1;
        }
    }
    out.push(&segs[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_path() {
        let s = parse_path("m 1,1 2,0 0,2 z m 10,10 h 1 v 1 z");
        assert_eq!(s.len(), 6);
        assert_eq!(s[0].start, [1.0, 1.0]);
        assert_eq!(s[1].end, [3.0, 3.0]);
        assert_eq!(s[2].end, [1.0, 1.0]);
        // after z the current point is the subpath start (1,1)
        assert_eq!(s[3].start, [11.0, 11.0]);
        let sub = continuous_subpaths(&s);
        assert_eq!(sub.len(), 2);
    }

    #[test]
    fn exponent_numbers() {
        let s = parse_path("M 1e-1,2 L 3.5e1 -4");
        assert_eq!(s[0].start, [0.1, 2.0]);
        assert_eq!(s[0].end, [35.0, -4.0]);
    }
}

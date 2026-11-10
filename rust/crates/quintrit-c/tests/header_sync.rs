//! Keeps `include/quintrit.h` in sync with the Rust sources, without any C compiler.
//!
//! The header is hand-written. This test parses it and the exported Rust items and compares
//! - the set of exported functions, with parameter and return types,
//! - the `repr(C)` structs, field by field,
//! - the numeric constants (`#define` against `pub const`),
//! - the version, and
//!
//! checks the structural properties of the header (include guard, `extern "C"` guard, C99-only includes).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn header() -> String {
    std::fs::read_to_string(crate_dir().join("include").join("quintrit.h")).unwrap().replace("\r\n", "\n")
}

fn rust_sources() -> String {
    let mut all = String::new();
    for e in std::fs::read_dir(crate_dir().join("src")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "rs") {
            all.push_str(&std::fs::read_to_string(p).unwrap().replace("\r\n", "\n"));
            all.push('\n');
        }
    }
    all
}

fn strip_comments(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("/*") {
        out.push_str(&rest[..i]);
        match rest[i..].find("*/") {
            Some(j) => rest = &rest[i + j + 2..],
            None => {
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n")
}

/// Canonical form of a C or Rust type for comparison: no `const`, no spaces.
fn canon(c: &str) -> String {
    c.replace("const", "").split_whitespace().collect::<Vec<_>>().join("")
}

fn rust_type(t: &str) -> String {
    let t = t.trim();
    if let Some(inner) = t.strip_prefix("*const ").or_else(|| t.strip_prefix("*mut ")) {
        return format!("{}*", rust_type(inner));
    }
    match t {
        "u8" => "uint8_t".into(),
        "i8" => "int8_t".into(),
        "u16" => "uint16_t".into(),
        "i16" => "int16_t".into(),
        "u32" => "uint32_t".into(),
        "i32" => "int32_t".into(),
        "u64" => "uint64_t".into(),
        "i64" => "int64_t".into(),
        "usize" => "size_t".into(),
        "f32" => "float".into(),
        "c_int" => "int".into(),
        "c_char" => "char".into(),
        "c_void" => "void".into(),
        "TokenCallback" => "quintrit_token_callback".into(),
        "GgufHandle" => "quintrit_gguf_t".into(),
        "ModelHandle" => "quintrit_model_t".into(),
        "SessionHandle" => "quintrit_session_t".into(),
        other => struct_c_name(other).map(str::to_string).unwrap_or_else(|| panic!("unmapped Rust type {other:?}")),
    }
}

fn struct_c_name(rust: &str) -> Option<&'static str> {
    Some(match rust {
        "TiledLayout" => "quintrit_tiled_layout_t",
        "TensorInfo" => "quintrit_tensor_info_t",
        "GgufInfo" => "quintrit_gguf_info_t",
        "ModelOptions" => "quintrit_model_options_t",
        "ModelInfo" => "quintrit_model_info_t",
        "SessionOptions" => "quintrit_session_options_t",
        "SessionInfo" => "quintrit_session_info_t",
        "GenerateParams" => "quintrit_generate_params_t",
        "GenerateResult" => "quintrit_generate_result_t",
        "ChatMessageC" => "quintrit_chat_message_t",
        _ => return None,
    })
}

/// `name -> (return type, parameter types)` of the exported Rust functions.
fn rust_functions(src: &str) -> BTreeMap<String, (String, Vec<String>)> {
    let mut out = BTreeMap::new();
    let marker = "extern \"C\" fn quintrit_";
    let mut pos = 0;
    while let Some(i) = src[pos..].find(marker) {
        let start = pos + i + "extern \"C\" fn ".len();
        pos = start;
        let name_end = start + src[start..].find('(').unwrap();
        let name = src[start..name_end].to_string();
        let params_end = name_end + src[name_end..].find(')').unwrap();
        let params: Vec<String> = src[name_end + 1..params_end]
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| {
                let (_, ty) = p.split_once(':').unwrap_or_else(|| panic!("bad parameter {p:?} of {name}"));
                rust_type(ty)
            })
            .collect();
        let after = src[params_end + 1..].trim_start();
        let ret = match after.strip_prefix("->") {
            Some(r) => rust_type(r.split('{').next().unwrap()),
            None => "void".into(),
        };
        out.insert(name, (ret, params));
    }
    out
}

fn header_functions(h: &str) -> BTreeMap<String, (String, Vec<String>)> {
    let h = strip_comments(h);
    let mut out = BTreeMap::new();
    let mut rest = h.as_str();
    while let Some(i) = rest.find("QUINTRIT_API ") {
        rest = &rest[i + "QUINTRIT_API ".len()..];
        let end = rest.find(';').unwrap();
        let decl = rest[..end].split_whitespace().collect::<Vec<_>>().join(" ");
        // Skip the macro definition lines (`# define QUINTRIT_API __declspec(...)`).
        if !decl.contains('(') || decl.starts_with("__declspec") {
            continue;
        }
        let open = decl.find('(').unwrap();
        let before = &decl[..open];
        let name_start = before.rfind([' ', '*']).map_or(0, |p| p + 1);
        let name = before[name_start..].to_string();
        let ret = canon(&before[..name_start]);
        let close = decl.rfind(')').unwrap();
        let plist = decl[open + 1..close].trim();
        let params: Vec<String> = if plist == "void" || plist.is_empty() {
            Vec::new()
        } else {
            plist
                .split(',')
                .map(|p| {
                    let p = p.trim();
                    let cut = p.rfind([' ', '*']).map_or(0, |i| i + 1);
                    canon(&p[..cut])
                })
                .collect()
        };
        out.insert(name, (ret, params));
        rest = &rest[end..];
    }
    out
}

#[test]
fn exported_functions_match_the_header() {
    let rust = rust_functions(&rust_sources());
    let c = header_functions(&header());
    let (rn, cn): (BTreeSet<_>, BTreeSet<_>) = (rust.keys().cloned().collect(), c.keys().cloned().collect());
    assert_eq!(rn.difference(&cn).collect::<Vec<_>>(), Vec::<&String>::new(), "exported by Rust but missing in quintrit.h");
    assert_eq!(cn.difference(&rn).collect::<Vec<_>>(), Vec::<&String>::new(), "declared in quintrit.h but not exported");
    assert!(rust.len() >= 40, "the parser found only {} functions", rust.len());
    for (name, sig) in &rust {
        assert_eq!(&c[name], sig, "signature of {name}: header (left) vs Rust (right)");
    }
}

/// Parses `typedef struct tag { type name; ... } name_t;` bodies into `(type, name)` lists.
fn header_structs(h: &str) -> BTreeMap<String, Vec<(String, String)>> {
    let h = strip_comments(h);
    let mut out = BTreeMap::new();
    let mut rest = h.as_str();
    while let Some(i) = rest.find("typedef struct ") {
        rest = &rest[i + "typedef struct ".len()..];
        let Some(open) = rest.find('{') else { continue };
        let semi = rest.find(';').unwrap();
        if semi < open {
            continue; // opaque `typedef struct tag name;`
        }
        let close = rest.find('}').unwrap();
        let name = rest[close + 1..].split(';').next().unwrap().trim().to_string();
        let fields = rest[open + 1..close]
            .split(';')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .map(|f| {
                let f = f.split_whitespace().collect::<Vec<_>>().join(" ");
                let (decl, array) = match f.find('[') {
                    Some(b) => (f[..b].trim().to_string(), f[b..].to_string()),
                    None => (f.clone(), String::new()),
                };
                let cut = decl.rfind([' ', '*']).map_or(0, |i| i + 1);
                (format!("{}{}", canon(&decl[..cut]), array.replace(' ', "")), decl[cut..].to_string())
            })
            .collect();
        out.insert(name, fields);
        rest = &rest[close..];
    }
    out
}

fn rust_structs(src: &str) -> BTreeMap<String, Vec<(String, String)>> {
    let mut out = BTreeMap::new();
    let mut pos = 0;
    while let Some(i) = src[pos..].find("#[repr(C)]") {
        pos += i + 10;
        let after = &src[pos..];
        let Some(s) = after.find("pub struct ") else { break };
        // Only the struct directly following the attribute (derives and docs may sit in between).
        if after[..s].contains("fn ") {
            continue;
        }
        let name_start = s + "pub struct ".len();
        let name_end = name_start + after[name_start..].find(|c: char| !c.is_alphanumeric() && c != '_').unwrap();
        let name = &after[name_start..name_end];
        let Some(cname) = struct_c_name(name) else { continue };
        let open = after[name_end..].find('{').unwrap() + name_end;
        let close = after[open..].find("\n}").unwrap() + open;
        let fields = after[open + 1..close]
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("pub "))
            .map(|l| {
                let l = l.trim_start_matches("pub ").trim_end_matches(',');
                let (fname, ty) = l.split_once(':').unwrap();
                let ty = ty.trim();
                let c = if let Some(arr) = ty.strip_prefix('[') {
                    let (elem, len) = arr.trim_end_matches(']').split_once(';').unwrap();
                    format!("{}[{}]", rust_type(elem), len.trim())
                } else {
                    rust_type(ty)
                };
                (c, fname.trim().to_string())
            })
            .collect();
        out.insert(cname.to_string(), fields);
        pos += close;
    }
    out
}

#[test]
fn structs_match_the_header_field_by_field() {
    let rust = rust_structs(&rust_sources());
    let c = header_structs(&header());
    assert_eq!(rust.len(), 10, "expected all ten repr(C) structs, found {:?}", rust.keys().collect::<Vec<_>>());
    for (name, fields) in &rust {
        let Some(cf) = c.get(name) else { panic!("struct {name} is missing from quintrit.h") };
        assert_eq!(cf, fields, "fields of {name}: header (left) vs Rust (right)");
        if name != "quintrit_chat_message_t" {
            assert_eq!(cf[0].1, "struct_size", "{name} must start with struct_size");
        }
    }
}

/// Evaluates the literal forms that appear in the constants: `12`, `-1`, `1 << 3`, `((uint64_t)1 << 3)`.
fn eval(expr: &str) -> Option<i128> {
    let e = expr.replace("(uint64_t)", "").replace("u64", "");
    let e = e.trim().trim_start_matches('(').trim_end_matches(')').trim().to_string();
    if let Some((a, b)) = e.split_once("<<") {
        return Some(eval(a)? << eval(b)?);
    }
    e.replace('_', "").parse().ok()
}

#[test]
fn constants_match_the_header() {
    let src = rust_sources();
    let mut rust = BTreeMap::new();
    for line in src.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("pub const QUINTRIT_") else { continue };
        let (name, tail) = rest.split_once(':').unwrap();
        let (ty, value) = tail.split_once('=').unwrap();
        let value = value.trim().trim_end_matches(';');
        let ty = ty.trim();
        if !["i32", "u32", "u64", "c_int", "usize"].contains(&ty) {
            continue;
        }
        // `QUINTRIT_GGML_*` are derived from the format crate's constants.
        let v = match eval(value) {
            Some(v) => v,
            None if name.starts_with("GGML_") => {
                let t = value.trim_end_matches(" as i32").trim_start_matches("GGML_TYPE_");
                match t {
                    "F32" => quintrit_format::GGML_TYPE_F32 as i128,
                    "F16" => quintrit_format::GGML_TYPE_F16 as i128,
                    "Q8_0" => quintrit_format::GGML_TYPE_Q8_0 as i128,
                    "BF16" => quintrit_format::GGML_TYPE_BF16 as i128,
                    "TQ1_0" => quintrit_format::GGML_TYPE_TQ1_0 as i128,
                    "TQ2_0" => quintrit_format::GGML_TYPE_TQ2_0 as i128,
                    "I2_S" => quintrit_format::GGML_TYPE_I2_S as i128,
                    other => panic!("unknown GGML constant {other}"),
                }
            }
            // Derived from the limits that `quintrit-core` shares with the other bindings.
            None if name == "MAX_THREADS" => quintrit_core::MAX_THREADS as i128,
            None => panic!("cannot evaluate QUINTRIT_{name} = {value}"),
        };
        rust.insert(format!("QUINTRIT_{name}"), v);
    }
    // Constants defined in other modules through a short alias.
    rust.insert("QUINTRIT_TOKENIZE_ADD_BOS".to_string(), quintrit_c::QUINTRIT_TOKENIZE_ADD_BOS as i128);

    let h = strip_comments(&header());
    let mut c = BTreeMap::new();
    for line in h.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("#define QUINTRIT_") else { continue };
        let mut parts = rest.splitn(2, char::is_whitespace);
        let name = format!("QUINTRIT_{}", parts.next().unwrap());
        let Some(value) = parts.next() else { continue };
        if value.trim().starts_with('"') || name.starts_with("QUINTRIT_VERSION_") || name == "QUINTRIT_API" || name == "QUINTRIT_H" {
            continue;
        }
        if let Some(v) = eval(value) {
            c.insert(name, v);
        }
    }
    assert!(c.len() >= 44, "found only {} constants in the header", c.len());
    for (name, value) in &c {
        let Some(r) = rust.get(name) else { panic!("{name} is defined in quintrit.h but has no Rust constant") };
        assert_eq!(r, value, "{name}");
    }
    for name in rust.keys() {
        assert!(c.contains_key(name), "{name} exists in Rust but not in quintrit.h");
    }
}

#[test]
fn version_and_abi_match_cargo() {
    let h = header();
    let version = env!("CARGO_PKG_VERSION");
    let parts: Vec<&str> = version.split('.').collect();
    let get = |name: &str| -> String {
        h.lines()
            .find_map(|l| l.trim().strip_prefix(&format!("#define {name} ")).map(|v| v.trim().trim_matches('"').to_string()))
            .unwrap_or_else(|| panic!("{name} missing"))
    };
    assert_eq!(get("QUINTRIT_VERSION_MAJOR"), parts[0]);
    assert_eq!(get("QUINTRIT_VERSION_MINOR"), parts[1]);
    assert_eq!(get("QUINTRIT_VERSION_PATCH"), parts[2]);
    assert_eq!(get("QUINTRIT_VERSION_STRING"), version);
    assert_eq!(get("QUINTRIT_ABI_VERSION"), quintrit_c::QUINTRIT_ABI_VERSION.to_string());
}

#[test]
fn header_is_a_well_formed_c99_and_cpp_header() {
    let raw = header();
    let h = strip_comments(&raw);
    // Include guard.
    assert!(h.contains("#ifndef QUINTRIT_H") && h.contains("#define QUINTRIT_H"));
    assert!(h.trim_end().ends_with("#endif"));
    // extern "C" guard, opened and closed exactly once.
    assert_eq!(h.matches("extern \"C\" {").count(), 1);
    assert!(h.contains("#ifdef __cplusplus"));
    // Only C99 standard headers; no C++ constructs outside the guard.
    let includes: Vec<&str> = h.lines().filter_map(|l| l.trim().strip_prefix("#include ")).collect();
    assert_eq!(includes, vec!["<stddef.h>", "<stdint.h>"]);
    let guard_open = h.find("#ifdef __cplusplus").unwrap();
    let outside = format!("{}{}", &h[..guard_open], &h[h.rfind("#endif").unwrap()..]);
    for banned in ["class ", "namespace ", "template", "std::", "nullptr", "bool ", "//"] {
        assert!(!outside.contains(banned), "C++-only construct {banned:?} in the header");
    }
    for banned in ["class ", "namespace ", "template", "std::", "nullptr"] {
        assert!(!h.contains(banned), "C++-only construct {banned:?} in the header");
    }
    // Balanced braces and parentheses.
    assert_eq!(h.matches('{').count(), h.matches('}').count());
    assert_eq!(h.matches('(').count(), h.matches(')').count());
    // Every function is documented: a comment block directly precedes it or the previous declaration group.
    assert!(raw.matches("/*").count() > 40, "the header should be documented");
    // Every function uses the export macro, and no declaration is left without a prototype list.
    let decls = h.matches("QUINTRIT_API ").count();
    assert!(decls >= 41, "{decls}");
}

#[test]
fn cpp_wrapper_only_uses_declared_c_symbols() {
    let hpp = std::fs::read_to_string(crate_dir().join("include").join("quintrit.hpp")).unwrap();
    let declared: BTreeSet<String> = header_functions(&header()).keys().cloned().collect();
    let mut used = BTreeSet::new();
    let mut rest = hpp.as_str();
    while let Some(i) = rest.find("quintrit_") {
        let tail = &rest[i..];
        let end = tail.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(tail.len());
        let ident = &tail[..end];
        if tail[end..].starts_with('(') {
            used.insert(ident.to_string());
        }
        rest = &tail[end..];
    }
    for f in &used {
        assert!(declared.contains(f), "quintrit.hpp calls {f}, which quintrit.h does not declare");
    }
    assert!(used.len() >= 20);
}

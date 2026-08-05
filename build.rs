use std::env;
use std::fs;
use time::{format_description, OffsetDateTime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = format_description::parse_borrowed::<2>("[year repr:last_two][month][day][hour][minute]")?;
    let dt = OffsetDateTime::now_utc().format(&format)?;
    println!("cargo:rustc-env=PACKAGE_COMPILE_TIME={}", dt);

    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=src/keccakf1600_x86-64.s");
    tonic_build::configure()
        .build_server(false)
        // .type_attribute(".", "#[derive(Debug)]")
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;
    // PoM mining kernel → PTX set (loaded at runtime into the miner's CUDA context).
    // Build a fallback ladder so mixed/older rigs don't fail on a single too-new .target.
    println!("cargo:rerun-if-changed=cuda/pom_mine.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=POM_FATBIN_LEGACY");
    println!("cargo:rerun-if-env-changed=POM_FATBIN_NEXTGEN");
    println!("cargo:rerun-if-changed=cuda/pom_mine_legacy.fatbin");
    println!("cargo:rerun-if-changed=cuda/pom_mine_nextgen.fatbin");
    let nvcc = env::var("NVCC").ok().unwrap_or_else(|| {
        let pinned = "/home/slash/cuda-12.2/bin/nvcc";
        if std::path::Path::new(pinned).exists() { pinned.to_string() } else { "nvcc".to_string() }
    });
    {
        let out_dir = env::var("OUT_DIR").unwrap();
        for sm in ["90", "89", "86", "80", "75", "70", "61"] {
            let ptx = format!("{out_dir}/pom_mine_sm{sm}.ptx");
            let output = std::process::Command::new(&nvcc)
                .args(["-ptx", "-O3", &format!("-arch=sm_{sm}"), "cuda/pom_mine.cu", "-o", &ptx])
                .output()
                .unwrap_or_else(|e| panic!("nvcc ({nvcc}) failed to run for sm_{sm}: {e}"));
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!(
                    "nvcc failed to compile cuda/pom_mine.cu for sm_{sm}:\n{}",
                    stderr
                );
            }
        }

        // Optional prebuilt fatbins: staged into OUT_DIR so runtime can embed/package them now,
        // then start loading them in a follow-up refactor.
        let legacy_src = env::var("POM_FATBIN_LEGACY")
            .unwrap_or_else(|_| "cuda/pom_mine_legacy.fatbin".to_string());
        let nextgen_src = env::var("POM_FATBIN_NEXTGEN")
            .unwrap_or_else(|_| "cuda/pom_mine_nextgen.fatbin".to_string());
        let legacy_dst = format!("{out_dir}/pom_mine_legacy.fatbin");
        let nextgen_dst = format!("{out_dir}/pom_mine_nextgen.fatbin");

        if std::path::Path::new(&legacy_src).exists() {
            fs::copy(&legacy_src, &legacy_dst).unwrap_or_else(|e| {
                panic!("failed copying POM_FATBIN_LEGACY from {legacy_src} to {legacy_dst}: {e}")
            });
        } else {
            fs::write(&legacy_dst, []).unwrap_or_else(|e| {
                panic!("failed creating empty legacy fatbin placeholder {legacy_dst}: {e}")
            });
        }

        if std::path::Path::new(&nextgen_src).exists() {
            fs::copy(&nextgen_src, &nextgen_dst).unwrap_or_else(|e| {
                panic!("failed copying POM_FATBIN_NEXTGEN from {nextgen_src} to {nextgen_dst}: {e}")
            });
        } else {
            fs::write(&nextgen_dst, []).unwrap_or_else(|e| {
                panic!("failed creating empty nextgen fatbin placeholder {nextgen_dst}: {e}")
            });
        }
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_arch == "x86_64" && target_os != "windows" && target_os != "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64.s").compile("libkeccak.a");
    }
    if target_arch == "x86_64" && target_os == "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64-osx.s").compile("libkeccak.a");
    }

    // In-process llama.cpp engine: build `libkeryx-llama.so` next to the miner binary.
    // llama.cpp is PINNED to b10015 — the SAME pin as hiveos/build-keryx-llama.sh and the
    // byte-identity proof in tools/llama_zerodup_spike. Bump all together, then re-verify.
    println!("cargo:rerun-if-changed=tools/keryx-llama/keryx_llama.cpp");
    println!("cargo:rerun-if-env-changed=KERYX_LLAMA_SKIP");
    println!("cargo:rerun-if-env-changed=KERYX_LLAMA_SRC");
    println!("cargo:rerun-if-env-changed=KERYX_LLAMA_ARCHS");
    if env::var("KERYX_LLAMA_SKIP").as_deref() == Ok("1") {
        println!("cargo:warning=KERYX_LLAMA_SKIP=1 — libkeryx-llama.so not built; a prebuilt one must sit next to the miner binary or in-process llama tiers cannot be mined");
    } else if target_arch == "x86_64" && (target_os == "linux" || target_os == "windows") {
        build_keryx_llama(&nvcc)?;
    }
    Ok(())
}

const LLAMA_TAG: &str = "b10015";
const LLAMA_ESCAPE_HINT: &str = "set KERYX_LLAMA_SKIP=1 to build the miner without it (a prebuilt libkeryx-llama.so must then be placed next to the binary)";

/// Builds `libkeryx-llama.so` into the cargo profile dir (next to the miner binary):
/// clones llama.cpp at `LLAMA_TAG` (cached under target/), builds its static libs via
/// cmake (incremental — near no-op on rebuilds), then links the keryx wrapper.
fn build_keryx_llama(nvcc: &str) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(env::var("OUT_DIR")?);
    // OUT_DIR = target/<profile>/build/<crate>-<hash>/out
    let profile_dir = out_dir.ancestors().nth(3).ok_or("cannot locate cargo profile dir from OUT_DIR")?;
    let target_root = profile_dir.parent().ok_or("cannot locate cargo target dir")?;

    let src = env::var("KERYX_LLAMA_SRC")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| target_root.join(format!("llama-src-{LLAMA_TAG}")));
    if !src.join("CMakeLists.txt").exists() {
        // Informational only — goes to the build-script log, not the user's terminal
        // (a cargo:warning here made every first build look broken).
        eprintln!("cloning llama.cpp {LLAMA_TAG} (first build only; use KERYX_LLAMA_SRC for an offline checkout)");
        run(
            "git clone of llama.cpp",
            std::process::Command::new("git").args([
                "clone", "--quiet", "--depth", "1", "--branch", LLAMA_TAG,
                "https://github.com/ggml-org/llama.cpp",
            ]).arg(&src),
        )?;
    }

    let build_dir = target_root.join(format!("llama-build-{LLAMA_TAG}"));
    // Ship real kernels for common GPUs plus compute_89 PTX, which drivers JIT-forward
    // to newer architectures such as Blackwell. Override for machine-specific builds.
    let archs = env::var("KERYX_LLAMA_ARCHS")
        .unwrap_or_else(|_| "75-real;80-real;86-real;89-real;89-virtual".to_string());
    let mut cfg = std::process::Command::new("cmake");
    cfg.arg("-S").arg(&src)
        .arg("-B").arg(&build_dir)
        .args([
            "-DGGML_CUDA=ON",
            &format!("-DCMAKE_CUDA_ARCHITECTURES={archs}"),
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DLLAMA_CURL=OFF",
            "-DGGML_NATIVE=OFF",
            // No-AVX CPU baseline (mirrors hiveos/build-keryx-llama.sh): BMI2/AVX code
            // traps with SIGILL on pre-Haswell CPUs, and the CPU path is only a fallback —
            // GPU inference is unaffected by these flags.
            "-DGGML_AVX=OFF",
            "-DGGML_AVX2=OFF",
            "-DGGML_FMA=OFF",
            "-DGGML_F16C=OFF",
            "-DGGML_BMI2=OFF",
            "-DGGML_CUDA_NCCL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            &format!("-DCMAKE_CUDA_COMPILER={nvcc}"),
        ]);
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        // -march=nehalem pins a no-AVX/no-BMI ISA baseline for everything the GGML guards
        // miss (and for toolchains whose default baseline is above x86-64). MSVC has no
        // -march and defaults to SSE2, so Windows needs no pin.
        cfg.args(["-DCMAKE_C_FLAGS=-march=nehalem", "-DCMAKE_CXX_FLAGS=-march=nehalem"]);
    }
    run(
        "cmake configure of llama.cpp (if it cannot detect a GPU, set KERYX_LLAMA_ARCHS explicitly)",
        &mut cfg,
    )?;
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "8".to_string());
    run(
        "cmake build of llama.cpp static libs",
        std::process::Command::new("cmake")
            .arg("--build").arg(&build_dir)
            // --config is required by MSVC's multi-config generator; single-config
            // generators (Makefiles/Ninja on Linux) silently ignore it.
            .args(["--target", "llama", "--config", "Release", "-j", &jobs]),
    )?;

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // MSVC multi-config layout: static libs land under <dir>/Release/*.lib. Link the
        // wrapper via nvcc (drives cl.exe and knows the CUDA include/lib paths; links
        // cudart statically by default). /openmp pulls vcomp for ggml-cpu's OpenMP use.
        let dll = profile_dir.join("keryx-llama.dll");
        let lib = |multi_config: &str, single_config: &str| {
            let path = build_dir.join(multi_config);
            if path.exists() { path } else { build_dir.join(single_config) }.into_os_string()
        };
        run(
            "link of keryx-llama.dll",
            std::process::Command::new(nvcc)
                .args(["-O2", "-std=c++17", "-shared", "tools/keryx-llama/keryx_llama.cpp"])
                // /MD is required: the llama.cpp static libs are built by CMake against the
                // dynamic CRT, and cl's /MT default breaks the link (LNK2001 __imp_modff etc.)
                .args(["-Xcompiler", "/EHsc", "-Xcompiler", "/openmp", "-Xcompiler", "/utf-8", "-Xcompiler", "/MD"])
                .arg("-I").arg(src.join("include"))
                .arg("-I").arg(src.join("ggml/include"))
                .arg("-I").arg(src.join("src"))
                .arg("-I").arg(src.join("common"))
                .arg(lib("src/Release/llama.lib", "src/llama.lib"))
                .arg(lib("ggml/src/ggml-cuda/Release/ggml-cuda.lib", "ggml/src/ggml-cuda/ggml-cuda.lib"))
                .arg(lib("ggml/src/Release/ggml-cpu.lib", "ggml/src/ggml-cpu.lib"))
                .arg(lib("ggml/src/Release/ggml.lib", "ggml/src/ggml.lib"))
                .arg(lib("ggml/src/Release/ggml-base.lib", "ggml/src/ggml-base.lib"))
                .args(["-lcublas", "-lcublasLt", "-lcuda", "-ladvapi32"])
                .arg("-o").arg(&dll),
        )?;
        return Ok(());
    }

    let cuda_home = cuda_home_from_nvcc(nvcc)?;
    let so = profile_dir.join("libkeryx-llama.so");
    let lib = |p: &str| build_dir.join(p).into_os_string();
    run(
        "link of libkeryx-llama.so",
        std::process::Command::new("g++")
            // -march=nehalem: same no-AVX/no-BMI baseline as the static libs (WRAPFLAGS in
            // hiveos/build-keryx-llama.sh) so the wrapper itself cannot SIGILL pre-Haswell.
            .args(["-O2", "-std=c++17", "-march=nehalem", "-shared", "-fPIC", "-fopenmp", "tools/keryx-llama/keryx_llama.cpp"])
            .arg("-I").arg(src.join("include"))
            .arg("-I").arg(src.join("ggml/include"))
            .arg("-I").arg(src.join("src"))
            .arg("-I").arg(src.join("common"))
            .arg("-I").arg(cuda_home.join("include"))
            .arg("-Wl,--start-group")
            .arg(lib("src/libllama.a"))
            .arg(lib("ggml/src/ggml-cuda/libggml-cuda.a"))
            .arg(lib("ggml/src/libggml-cpu.a"))
            .arg(lib("ggml/src/libggml.a"))
            .arg(lib("ggml/src/libggml-base.a"))
            .arg("-Wl,--end-group")
            .arg(format!("-L{}", cuda_home.join("lib64").display()))
            .arg(format!("-L{}", cuda_home.join("targets/x86_64-linux/lib").display()))
            .args(["-lcudart", "-lcublas", "-lcublasLt"])
            .arg(format!("-L{}", cuda_home.join("lib64/stubs").display()))
            .arg(format!("-L{}", cuda_home.join("targets/x86_64-linux/lib/stubs").display()))
            .args(["-lcuda", "-lpthread", "-ldl"])
            .arg("-o").arg(&so),
    )?;
    Ok(())
}

/// Resolves the CUDA toolkit root for include/lib paths from the nvcc in use.
fn cuda_home_from_nvcc(nvcc: &str) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let p = std::path::Path::new(nvcc);
    if let Some(root) = p.parent().and_then(|bin| bin.parent()) {
        if root.components().count() > 0 && root.join("include").exists() {
            return Ok(root.to_path_buf());
        }
    }
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(v) = env::var(var) {
            return Ok(std::path::PathBuf::from(v));
        }
    }
    let default = std::path::Path::new("/usr/local/cuda");
    if default.exists() {
        return Ok(default.to_path_buf());
    }
    Err(format!("cannot locate the CUDA toolkit root (needed to link libkeryx-llama.so); set CUDA_HOME, or {LLAMA_ESCAPE_HINT}").into())
}

fn run(desc: &str, cmd: &mut std::process::Command) -> Result<(), Box<dyn std::error::Error>> {
    let status = cmd.status().map_err(|e| format!("{desc}: failed to launch: {e} — {LLAMA_ESCAPE_HINT}"))?;
    if !status.success() {
        return Err(format!("{desc} failed ({status}) — {LLAMA_ESCAPE_HINT}").into());
    }
    Ok(())
}

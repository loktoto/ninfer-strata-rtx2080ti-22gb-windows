# Windows development test package

This creates a **development_unqualified** ZIP for testing an existing Windows build. The packager has not been validated by running a Windows-built archive on a target machine. It does not build, execute, download, or qualify a binary or model. A compatible NVIDIA GPU and driver are required for the engine and GPU parity tests; driver-only operation of the packaged binaries remains to be tested.

From a native Windows x64 Visual Studio developer shell with `dumpbin` on PATH, build the desired targets first, for example:

```powershell
.\tools\build_windows.ps1
cmake --build build-windows --target sage_rmsnorm_parity sage_q4k_slots_parity sage_q6k_slots_parity sage_mma_probe
python tools\package_windows.py --build-dir build-windows --output .\strata-test.zip `
  --dll-dir "$env:CUDA_PATH\bin" `
  --cuda-license "C:\path\to\official\NVIDIA-license.txt" `
  --license-file "C:\path\to\other-dependency-license.txt"
```

Adjust the license paths to the official files applicable to your exact toolkit and dependencies. The packager requires `--cuda-license` if it identifies CUDA DLLs, but cannot determine whether the supplied license grants redistribution rights. Review the applicable NVIDIA and other dependency terms yourself. Supply every applicable third-party license using repeatable `--license-file`; the project LICENSE and NOTICE are always included. The tool does not fetch licenses or certify redistribution rights. Point repeatable `--dll-dir` options at any other runtime directories needed, including Visual C++ runtime files if they are to be bundled.

By default the tool searches the specified build tree for `strata-poly.exe`, `sage_rmsnorm_parity.exe`, `sage_q4k_slots_parity.exe`, `sage_q6k_slots_parity.exe`, and `sage_mma_probe.exe`. The engine is required; parity executables are included when found. Duplicate matches fail. To select exact build products, use repeatable `--binary` paths inside the build tree; this also supports additional targets such as a Q6 parity executable when that target exists. Explicit selections replace the defaults.

The packager reads `dumpbin /DEPENDENTS` for each selected binary and each bundled DLL recursively. It searches the selected binaries' directories and explicitly supplied DLL directories, and rejects missing dependencies and different files with the same DLL basename. A conservative list of Windows system DLLs, NVIDIA's driver DLL, and Windows API-set contracts are treated as target-system dependencies. DLLs loaded dynamically rather than appearing in the import table cannot be discovered by this method. The upstream mr-september packaging script at `8dfa5cc96620b646be4608ffb1246cb89df9a444` copies adjacent DLLs and one cudart DLL; recursive import inspection covers additional transitive imports but still requires target-machine validation.

The output must be a new ZIP path in an existing directory. Staging is temporary; existing output is never overwritten. The manifest records SHA-256 for every other archive file, root Git commit, dirty state, patch hashes, and source-lock hash. A dirty state means the commit alone does not fully identify source contents. It is a packaging manifest, not a signature or proof of the build's origin. `--dry-run` resolves and reports dependencies without creating a ZIP. It can run outside Windows only if a compatible `dumpbin` command is available; host-side mocked tests do not prove real PE behavior.

Extract the archive and run the included parity executables on the target GPU. For the Python API, prepare the repository and pinned PolyStrata checkout as described in [API.md](API.md), then run `python tools/serve_sage.py --engine <extracted-directory>\strata-poly.exe --model <local-model> --output-dir <new-directory>`. The Python API, its dependencies, and the model are not included in this archive. Follow the repository validation procedures before treating any result as qualified.

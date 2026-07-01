# tests/test_machineid.py
from tokenwatt.machineid import detect_machine, MachineInfo


def _fake_sysctl(mapping):
    return lambda key: mapping[key]


def test_detects_m3_ultra_and_builds_stable_slug():
    info = detect_machine(
        sysctl=_fake_sysctl({
            "machdep.cpu.brand_string": "Apple M3 Ultra",
            "hw.model": "Mac15,14",
            "hw.memsize": str(96 * 1024**3),
        }),
        mac_ver=lambda: "26.5.1")
    assert isinstance(info, MachineInfo)
    assert info.soc == "Apple M3 Ultra"
    assert info.model == "Mac15,14"
    assert info.ram_gb == 96
    assert info.macos_major == "26"
    # slug is stable, lowercase, filesystem-safe, and encodes the identity
    assert info.machine_id == "mac15-14_apple-m3-ultra_96gb_macos26"
    assert "M3 Ultra" in info.label and "96" in info.label


def test_slug_is_deterministic():
    kw = dict(sysctl=_fake_sysctl({"machdep.cpu.brand_string": "Apple M4",
                                   "hw.model": "Mac16,12", "hw.memsize": str(16 * 1024**3)}),
              mac_ver=lambda: "26.1")
    assert detect_machine(**kw).machine_id == detect_machine(**kw).machine_id
    assert detect_machine(**kw).machine_id == "mac16-12_apple-m4_16gb_macos26"

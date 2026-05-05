from __future__ import annotations

from scripts.system.mount_usb_readonly import is_usb_candidate


def test_usb_candidate_requires_removable_or_usb_transport():
    assert is_usb_candidate(
        {
            "type": "part",
            "fstype": "exfat",
            "rm": True,
            "tran": "usb",
            "path": "/dev/sdb1",
        }
    )
    assert not is_usb_candidate(
        {
            "type": "part",
            "fstype": "ext4",
            "rm": False,
            "tran": "nvme",
            "path": "/dev/nvme1n1p2",
        }
    )

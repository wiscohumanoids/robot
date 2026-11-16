"""The guard must fail closed before model construction on an occupied GPU."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infra.rl import resources


class ResourceTests(unittest.TestCase):
    def query(self, fields, kind="gpu"):
        if kind == "gpu":
            return [["0", "GPU-test", "Test GPU", "8192", "7000", "0"]]
        return []

    def test_foreign_compute_pid_rejected_even_with_free_memory(self):
        def occupied(fields, kind="gpu"):
            if kind == "compute-apps":
                return [["GPU-test", "999999", "current-trainer", "100"]]
            return self.query(fields, kind)
        with patch.dict(os.environ, {}, clear=True), patch.object(resources, "_query", side_effect=occupied):
            report = resources.assess_resources()
            self.assertFalse(report["allowed"])
            self.assertIn("999999", report["reasons"][0])
            with self.assertRaises(resources.ResourceError):
                with resources.require_idle_gpu():
                    self.fail("An occupied GPU was accepted")

    def test_own_context_allowed_but_low_memory_rejected(self):
        def own(fields, kind="gpu"):
            if kind == "compute-apps":
                return [["GPU-test", str(os.getpid()), "self", "100"]]
            return self.query(fields, kind)
        with patch.dict(os.environ, {}, clear=True), patch.object(resources, "_query", side_effect=own):
            self.assertTrue(resources.assess_resources()["allowed"])
            self.assertFalse(resources.assess_resources(minimum_free_gib=8)["allowed"])

    def test_lease_blocks_second_process_context_and_releases(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch.object(resources, "_query", side_effect=self.query):
            with resources.require_idle_gpu(lock_dir=directory) as report:
                self.assertTrue(Path(report["lock_path"]).exists())
                with self.assertRaisesRegex(resources.ResourceError, "already leased"):
                    with resources.require_idle_gpu(lock_dir=directory):
                        self.fail("Concurrent EXPO lease was accepted")
            with resources.require_idle_gpu(lock_dir=directory):
                pass

    def test_activity_appearing_after_initial_check_rejected(self):
        process_queries = 0
        def race(fields, kind="gpu"):
            nonlocal process_queries
            if kind == "compute-apps":
                process_queries += 1
                return [] if process_queries == 1 else [["GPU-test", "999999", "trainer", "100"]]
            return self.query(fields, kind)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch.object(resources, "_query", side_effect=race):
            with self.assertRaisesRegex(resources.ResourceError, "999999"):
                with resources.require_idle_gpu(lock_dir=directory):
                    self.fail("Resource race was accepted")

    def test_cpu_does_not_query_nvidia_and_disk_threshold_applies(self):
        with patch.object(resources, "_query", side_effect=AssertionError("CUDA query on CPU")):
            self.assertTrue(resources.assess_resources("cpu")["allowed"])
            self.assertFalse(resources.assess_resources("cpu", minimum_disk_gib=10**10)["allowed"])

    def test_query_failure_fails_closed_and_visible_index_remaps(self):
        with patch.object(resources, "_query", side_effect=resources.ResourceError("driver unavailable")):
            self.assertFalse(resources.assess_resources()["allowed"])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test"}, clear=True), \
                patch.object(resources, "_query", side_effect=self.query):
            self.assertTrue(resources.assess_resources()["allowed"])
            self.assertFalse(resources.assess_resources("cuda:1")["allowed"])

    def test_current_aws_snapshot_rejected_with_zero_memory_floor(self):
        def aws_snapshot(fields, kind="gpu"):
            if kind == "gpu":
                return [["0", "GPU-aws-snapshot", "RTX PRO 6000 Blackwell", "97887", "19156", "100"]]
            return [["GPU-aws-snapshot", "29790", "train_panthera_new50.py", "68018"],
                    ["GPU-aws-snapshot", "30967", "inference/server.py", "10150"]]
        with patch.dict(os.environ, {}, clear=True), patch.object(resources, "_query", side_effect=aws_snapshot):
            report = resources.assess_resources(minimum_free_gib=0)
            self.assertFalse(report["allowed"])
            self.assertIn("29790", report["reasons"][0])
            self.assertIn("30967", report["reasons"][0])
            with self.assertRaises(resources.ResourceError):
                with resources.require_idle_gpu(minimum_free_gib=0):
                    self.fail("Active AWS training was bypassed")


if __name__ == "__main__":
    unittest.main()

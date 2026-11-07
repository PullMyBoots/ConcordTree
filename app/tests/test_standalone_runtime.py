"""Runtime placement contracts needed when sibling experiments are deleted."""
from pathlib import Path
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import runtime_config


class StandaloneRuntime(unittest.TestCase):
    def test_dangling_input_links_remain_protected(self):
        # All fixtures live in release-local temporary storage, outside real data/.
        with tempfile.TemporaryDirectory(dir=HERE / '.runtime') as directory:
            project = Path(directory) / 'project'
            (project / 'data').mkdir(parents=True)
            missing = project / 'reserved-input' / 'missing.phy'
            (project / 'data' / 'unrelated').symlink_to(missing)
            source = project / 'existing.phy'
            source.write_text('owned fixture')
            try:
                with patch.object(runtime_config, 'PROJECT', project):
                    runtime_config.protected_targets.cache_clear()
                    self.assertIn(missing, runtime_config.protected_targets())
                    output = project / 'runs' / 'job'
                    self.assertEqual(runtime_config.owned_path(output, inputs=[source]), output)
                    for target in (missing, missing / 'child', project / 'data' / 'unrelated'):
                        with self.subTest(target=str(target)), self.assertRaises(ValueError):
                            runtime_config.owned_path(target)
                    with self.assertRaises(FileNotFoundError):
                        runtime_config.owned_path(output, inputs=[missing])
                    self.assertFalse(missing.exists())
                    self.assertEqual(source.read_text(), 'owned fixture')
            finally:
                runtime_config.protected_targets.cache_clear()

    def test_external_cache_override_stays_in_release(self):
        with patch.dict(os.environ, {'TRITON_CACHE_DIR': '/tmp/foreign-experiment-cache'}):
            result = runtime_config.configure_runtime(cpu_threads=2)
        self.assertTrue(all(Path(path).is_relative_to(HERE)
                            for path in result['caches'].values()))

    def test_absent_research_project_marker_uses_release_as_output_root(self):
        module = type(sys)('standalone_policy_probe')
        module.__file__ = str(HERE / 'runtime_config.py')
        original = Path.is_dir
        def exists(path):
            return False if path.name == '.ResearchProject' else original(path)
        with patch.object(Path, 'is_dir', exists):
            exec(compile((HERE / 'runtime_config.py').read_text(), module.__file__, 'exec'),
                 module.__dict__)
        self.assertEqual(module.PROJECT, HERE)
        self.assertEqual(module.owned_path(HERE / 'runs/probe'), HERE / 'runs/probe')
        with self.assertRaises(ValueError):
            module.owned_path(HERE.parent / 'foreign-output')


if __name__ == '__main__':
    unittest.main(verbosity=2)

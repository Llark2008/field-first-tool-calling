import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from toolcall_lab.run import runtime_root, configure_runtime


class RuntimeTests(unittest.TestCase):
    def test_runtime_cannot_be_inside_source_or_its_parent(self):
        source = Path('/private/tmp/toolcall-source')
        for value in (source, source / '.venv', source.parent):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime_root(source, str(value))

    def test_default_is_outside_source(self):
        source = Path(__file__).resolve().parents[1]
        value = runtime_root(source)
        self.assertFalse(value.is_relative_to(source))
        self.assertTrue(value.is_absolute())

    def test_symlink_into_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            source = parent / 'source'
            source.mkdir()
            alias = parent / 'alias'
            alias.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                runtime_root(source, str(alias / 'runtime'))

    def test_nested_runtime_symlinks_cannot_redirect_writes_into_source(self):
        for relative in ('cache', 'runs', 'tmp', 'venv', 'cache/huggingface/hub', 'cache/pip/wheels/redirect'):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as temp:
                source, root = Path(temp) / 'source', Path(temp) / 'runtime'
                source.mkdir()
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source, target_is_directory=True)
                with self.assertRaises(ValueError):
                    runtime_root(source, str(root))

    def test_python_entry_overrides_inherited_huggingface_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / 'runtime'
            with patch.dict(os.environ, {'HF_HUB_CACHE': str(Path(temp) / 'bad-hub'), 'HF_XET_CACHE': str(Path(temp) / 'bad-xet')}):
                configure_runtime(root)
                self.assertEqual(Path(os.environ['HF_HUB_CACHE']), root / 'cache/huggingface/hub')
                self.assertEqual(Path(os.environ['HF_XET_CACHE']), root / 'cache/huggingface/xet')


if __name__ == '__main__':
    unittest.main()

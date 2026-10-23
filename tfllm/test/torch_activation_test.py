"""Native vector activations: numerics, dtype, aliasing and automatic routes."""
import os
from concurrent.futures import ThreadPoolExecutor
import unittest
import warnings
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from tfllm.torch import NpuRuntime, optimize, restore, optimization_report, make_backend, Unsupported


class Activations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.runtime = NpuRuntime(backend=os.environ.get("TFLLM_TEST_BACKEND", "cpu"))

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def pairs(self):
        return [(self.runtime.gelu, F.gelu),
                (lambda x: self.runtime.gelu(x, approximate="tanh"), lambda x: F.gelu(x, approximate="tanh")),
                (self.runtime.silu, F.silu)]

    def close(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        # FP32 vector erf uses TFDL's A&S approximation, not the tanh GELU.
        torch.testing.assert_close(actual, expected, atol=2e-6 if actual.dtype == torch.float32 else 3e-4,
                                   rtol=2e-6 if actual.dtype == torch.float32 else 1e-3, equal_nan=True)

    def test_all_fp16_encodings_and_fp32_extremes(self):
        half = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        torch.manual_seed(280)
        fp32 = torch.cat([torch.linspace(-20, 20, 131071), torch.randn(10003)*3,
                          torch.tensor([float('nan'), float('inf'), -float('inf'), 0., -0., 1e-40, -1e-40,
                                        3e38, -3e38, 65536.125, -65536.125])])
        with torch.inference_mode():
            for native, reference in self.pairs():
                for x in (half, fp32):
                    with self.subTest(dtype=x.dtype, op=reference):
                        expected = reference(x)
                        # Some ARM Torch tanh-GELU kernels multiply x*2 before
                        # halving and overflow at +3e38. Check the mathematical
                        # finite result against FP64 there, not that overflow.
                        large = torch.isfinite(x) & (x.abs() > 1e30)
                        expected[large] = reference(x[large].double()).to(x.dtype)
                        self.close(native(x), expected)

    def test_shapes_strides_autocast_and_unchanged_input(self):
        with torch.inference_mode():
            for dtype in (torch.float16, torch.float32):
                base = torch.linspace(-5, 5, 143, dtype=dtype).reshape(11, 13)
                for x in (base.t(), base[:, ::2], base[2, 3], base[:0], base[2:3, 3:4].expand(4, 5)):
                    before = x.clone()
                    for native, reference in self.pairs():
                        with torch.autocast('cpu', dtype=torch.bfloat16):
                            self.close(native(x), reference(x))
                        self.assertTrue(torch.equal(x, before))
                # Prove FP32 never narrows through FP16 (overflow would be Inf).
                self.assertEqual(self.runtime.silu(torch.tensor([65536.125])).item(), 65536.125)

    def test_inplace_views_overlap_and_version_counter(self):
        x = torch.randn(17, 13)
        before = x.clone()
        version = x._version
        view = x[:, ::2]
        expected = F.silu(view)
        with torch.no_grad():
            self.assertIs(self.runtime.silu(view, inplace=True), view)
        self.close(view, expected)
        self.assertTrue(torch.equal(x[:, 1::2], before[:, 1::2]))
        self.assertGreater(x._version, version)
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'single memory location'):
            self.runtime.silu(torch.ones(1).expand(8), inplace=True)

    def test_autograd_unsupported_dtype_closed_and_bad_approximation(self):
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            self.runtime.gelu(torch.ones(8, requires_grad=True))
        with self.assertRaises(TypeError):
            self.runtime.silu(torch.ones(8, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            self.runtime.gelu(torch.ones(8), approximate='quick')
        closed = NpuRuntime(backend='cpu')
        closed.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            closed.gelu(torch.empty(0))

    def test_module_alias_hooks_attributes_restore_and_fallback(self):
        model = nn.Sequential(nn.GELU(), nn.SiLU()).eval()
        model.alias = model[0]
        keys = list(model.state_dict())
        hits = []
        hook = model[0].register_forward_hook(lambda *args: hits.append(1))
        optimize(model, runtime=self.runtime, activation_policy='native')
        self.assertIs(model.alias, model[0])
        self.assertEqual(keys, list(model.state_dict()))
        with torch.no_grad():
            x = torch.linspace(-4, 4, 809)
            self.close(model[0](input=x), F.gelu(x))
            model[0].approximate = 'tanh'
            self.close(model[0](x), F.gelu(x, approximate='tanh'))
            self.close(model[1](x), F.silu(x))
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(c['status'] == 'cpu_fast' for c in optimization_report(model).to_dict()['calls']))
        with self.assertRaisesRegex(Unsupported, 'no_grad'):
            model[0](x)
        restore(model)
        self.assertNotIn('forward', model[0].__dict__)
        hook.remove()
        model = nn.SiLU().eval()
        optimize(model, runtime=self.runtime, strict=False, activation_policy='native')
        x = torch.ones(8, requires_grad=True)
        with warnings.catch_warnings(record=True) as caught:
            model(x).sum().backward()
        self.assertTrue(caught)
        self.assertIsNotNone(x.grad)
        with torch.no_grad(), mock.patch.object(self.runtime, 'silu', side_effect=RuntimeError('native failure')):
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                model(x)
        restore(model)

    def test_predicate_custom_subclass_and_atomic_preflight(self):
        class Custom(nn.SiLU):
            pass
        model = nn.Sequential(nn.GELU(), Custom()).eval()
        with self.assertRaisesRegex(Unsupported, 'subclasses'):
            optimize(model, runtime=self.runtime)
        self.assertNotIn('forward', model[0].__dict__)
        optimize(model, runtime=self.runtime, predicate=lambda path, module: path != '1')
        self.assertNotIn('forward', model[1].__dict__)
        restore(model)

    def test_compile_functional_and_aten(self):
        def f(x):
            return F.silu(F.gelu(x)) + F.gelu(x, approximate='tanh')
        with torch.no_grad():
            backend = make_backend(runtime=self.runtime, activation_policy='native')
            compiled = torch.compile(f, backend=backend, fullgraph=True)
            for dtype in (torch.float16, torch.float32):
                x = torch.linspace(-5, 5, 173, dtype=dtype)
                self.close(compiled(x), f(x))
            self.assertEqual({e['operator'] for e in backend.report.entries}, {'gelu', 'silu'})
            self.assertTrue(all(c['status']=='cpu_fast' for c in backend.report.to_dict()['calls']))
            def inplace(x):
                return F.silu(x, True)
            compiled = torch.compile(inplace, backend=make_backend(runtime=self.runtime, activation_policy='native'), fullgraph=True)
            x = torch.randn(31, 3).t()
            expected = F.silu(x)
            self.assertIs(compiled(x), x)
            self.close(x, expected)
            def aten(x):
                return torch.ops.aten.silu.default(torch.ops.aten.gelu.default(x, approximate='none'))
            compiled = torch.compile(aten, backend=make_backend(runtime=self.runtime, activation_policy='native'), fullgraph=True)
            self.close(compiled(x), aten(x))

    def test_auto_policy_retains_fast_torch_paths(self):
        for policy in ('auto', 'native', 'torch'):
            model = nn.SiLU().eval()
            optimize(model, runtime=self.runtime, activation_policy=policy)
            with torch.no_grad(), mock.patch.object(self.runtime, 'silu', wraps=self.runtime.silu) as native:
                for dtype, n in ((torch.float32, 1048576), (torch.float16, 17), (torch.float16, 65536)):
                    x = torch.randn(n, dtype=dtype)
                    self.close(model(x), F.silu(x))
                self.assertEqual(native.call_count, {'auto':1, 'native':3, 'torch':0}[policy])
            restore(model)
        with self.assertRaisesRegex(ValueError, 'activation_policy'):
            optimize(nn.GELU().eval(), runtime=self.runtime, activation_policy='invalid')

    def test_concurrent_calls_share_workers_without_sharing_outputs(self):
        inputs = [torch.randn(65543, dtype=dtype) for dtype in (torch.float16, torch.float32) for _ in range(2)]
        def run(x):
            with torch.inference_mode():
                return self.runtime.silu(self.runtime.gelu(x))
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(run, inputs))
        for x, y in zip(inputs, outputs):
            self.close(y, F.silu(F.gelu(x)))
        self.assertEqual(len({y.data_ptr() for y in outputs}), len(outputs))


if __name__ == '__main__':
    unittest.main()

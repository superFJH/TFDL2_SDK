"""Extended unary activation kernels and parameter-preserving Torch routes."""
import os
import unittest
from unittest import mock
import torch
from torch import nn
from torch.nn import functional as F
from tfllm.torch import NpuRuntime, optimize, restore, optimization_report, make_backend, Unsupported

OPS = {'sigmoid': (nn.Sigmoid, torch.sigmoid), 'tanh': (nn.Tanh, torch.tanh), 'mish': (nn.Mish, F.mish),
       'softplus': (nn.Softplus, F.softplus), 'elu': (nn.ELU, F.elu), 'selu': (nn.SELU, F.selu)}


class ExtendedActivations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.runtime = NpuRuntime(backend=os.environ.get('TFLLM_TEST_BACKEND', 'cpu'))

    @classmethod
    def tearDownClass(cls):cls.runtime.close()

    def close(self, got, expected):
        self.assertEqual(got.dtype, expected.dtype)
        self.assertEqual(got.shape, expected.shape)
        torch.testing.assert_close(got, expected, rtol=1e-3 if got.dtype == torch.float16 else 2e-6,
                                   atol=6e-5 if got.dtype == torch.float16 else 2e-6, equal_nan=True)

    def test_exhaustive_half_dense_float_and_special_values(self):
        half = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        fp32 = torch.cat([torch.linspace(-20, 20, 131071), torch.tensor([0., -0., 1e-40, -1e-40, 1e-6, -1e-6,
            3e38, -3e38, float('inf'), -float('inf'), float('nan')])])
        with torch.inference_mode():
            for kind, (_, reference) in OPS.items():
                for x in (half, fp32):
                    with self.subTest(kind=kind, dtype=x.dtype):
                        self.close(getattr(self.runtime, kind)(x), reference(x))

    def test_shapes_layout_autocast_and_scalar_generic_entry(self):
        with torch.inference_mode():
            for kind, (_, reference) in OPS.items():
                for dtype in (torch.float16, torch.float32):
                    base = torch.linspace(-4, 4, 221, dtype=dtype).reshape(13, 17)
                    for x in (base.t(), base[:, ::2], base[:0], base[2, 3]):
                        before = x.clone()
                        with torch.autocast('cpu', dtype=torch.bfloat16):
                            self.close(self.runtime.activation(x, kind), reference(x))
                        self.assertTrue(torch.equal(before, x))
        with self.assertRaisesRegex(NotImplementedError, 'beta'):
            self.runtime.softplus(torch.ones(3), beta=2)
        with self.assertRaisesRegex(NotImplementedError, 'alpha'):
            self.runtime.elu(torch.ones(3), alpha=.5)
        with self.assertRaisesRegex(ValueError, 'unknown activation'):
            self.runtime.activation(torch.ones(3), 'custom')

    def test_module_routes_restore_and_mutable_parameters(self):
        with torch.no_grad():
            for kind, (cls, reference) in OPS.items():
                model = cls().eval()
                optimize(model, runtime=self.runtime, activation_policy='native')
                x = torch.randn(37, 19)
                self.close(model(x), reference(x))
                self.assertEqual(optimization_report(model).to_dict()['calls'][0]['status'], 'cpu_fast')
                restore(model)
                self.assertNotIn('forward', model.__dict__)
            for model in (nn.Softplus(beta=2, threshold=5), nn.ELU(alpha=.2)):
                model.eval()
                reference = model.forward
                optimize(model, runtime=self.runtime)
                x = torch.randn(65536, dtype=torch.float16)
                self.close(model(x), reference(x))
                self.assertTrue(all(c['status']=='cpu_torch' for c in optimization_report(model).to_dict()['calls']))
                if isinstance(model, nn.Softplus):model.beta, model.threshold = 1, 20
                else:model.alpha = 1
                self.close(model(x), reference(x))
                self.assertTrue(any(c['status']=='cpu_fast' for c in optimization_report(model).to_dict()['calls']))
                restore(model)
        with self.assertRaisesRegex(Unsupported, 'beta'):
            optimize(nn.Softplus(beta=2).eval(), runtime=self.runtime, activation_policy='native')

    def test_inplace_views_version_overlap_and_gradients(self):
        for kind in ('mish', 'elu', 'selu'):
            x = torch.randn(13, 17)
            original, version = x.clone(), x._version
            view = x[:, ::2]
            expected = OPS[kind][1](view)
            with torch.no_grad():self.assertIs(getattr(self.runtime, kind)(view, inplace=True), view)
            self.close(view, expected)
            self.assertTrue(torch.equal(x[:, 1::2], original[:, 1::2]))
            self.assertGreater(x._version, version)
            with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'single memory location'):
                getattr(self.runtime, kind)(torch.ones(1).expand(8), inplace=True)
        for kind in OPS:
            with self.assertRaisesRegex(RuntimeError, 'inference-only'):
                getattr(self.runtime, kind)(torch.ones(2, requires_grad=True))

    def test_measured_auto_gates(self):
        cases = [('sigmoid', torch.float32, 1048576, False), ('tanh', torch.float32, 65536, False),
                 ('tanh', torch.float32, 262144, True), ('mish', torch.float32, 4096, True),
                 ('softplus', torch.float32, 65536, True), ('elu', torch.float16, 17, False),
                 ('selu', torch.float16, 65536, True)]
        with torch.no_grad():
            for kind, dtype, n, native in cases:
                model = OPS[kind][0]().eval()
                optimize(model, runtime=self.runtime)
                with mock.patch.object(self.runtime, kind, wraps=getattr(self.runtime, kind)) as call:
                    x = torch.randn(n, dtype=dtype)
                    self.close(model(x), OPS[kind][1](x))
                    self.assertEqual(call.call_count, int(native))
                restore(model)

    def test_compiled_functional_aten_methods_and_parameter_fallback(self):
        def functions(x):
            return torch.sigmoid(x) + torch.tanh(x) + F.mish(x) + F.softplus(x) + F.elu(x) + F.selu(x)
        def aten(x):
            return (torch.ops.aten.sigmoid.default(x) + torch.ops.aten.tanh.default(x)
                    + torch.ops.aten.mish.default(x) + torch.ops.aten.softplus.default(x)
                    + torch.ops.aten.selu.default(x))
        def methods(x):return x.sigmoid() + x.tanh()
        with torch.no_grad():
            for function, operators in ((functions, set(OPS)), (aten, set(OPS)-{'elu'}), (methods, {'sigmoid', 'tanh'})):
                backend = make_backend(runtime=self.runtime, activation_policy='native')
                compiled = torch.compile(function, backend=backend, fullgraph=True)
                for dtype in (torch.float16, torch.float32):
                    x = torch.linspace(-4, 4, 313, dtype=dtype)
                    self.close(compiled(x), function(x))
                self.assertEqual({e['operator'] for e in backend.report.entries}, operators)
            def custom(x):return F.softplus(x, 2, 5) + F.elu(x, .2)
            backend = make_backend(runtime=self.runtime)
            compiled = torch.compile(custom, backend=backend, fullgraph=True)
            x = torch.randn(65536, dtype=torch.float16)
            self.close(compiled(x), custom(x))
            self.assertTrue(all(c['status']=='cpu_torch' for c in backend.report.to_dict()['calls']))


if __name__ == '__main__':unittest.main()

import unittest
from summarize_decode import summarize


class SummaryTest(unittest.TestCase):
    def test_interleaved_round_medians_and_greedy_rate(self):
        lines = []
        for rnd, factor in enumerate([1, 100, 2]):
            for arm in [0, 1]:
                for rep, ns in enumerate([5, 3, 4]):
                    lines.append(f'[perf-width] rows=2 round={rnd} rep={rep} arm={arm} ns={factor * ns * (arm + 1) * 1000000}')
                lines.append(f'[perf-greedy] round={rnd} arm={arm} tokens=200 ns={(arm + 1) * (rnd + 1) * 1000000000}')
        got = summarize('\n'.join(lines))
        self.assertEqual(got['width_ms']['2'], [8.0, 16.0])
        self.assertEqual(got['greedy_tok_s'], [100.0, 50.0])
        self.assertEqual(got['greedy_tokens_per_sample'], 200)

    def test_chain_samples(self):
        got = summarize('DECODE rows=1 round=0 rep=15 arm=0 ns=800000\nDECODE rows=1 round=0 rep=15 arm=1 ns=500000')
        self.assertEqual(got['width_ms']['1'], [0.8, 0.5])
        self.assertIsNone(got['greedy_identical'])

    def test_affine_formats_remain_separate(self):
        raw = 'AFFINE bits=2 gs=128 rows=1 round=0 rep=15 arm=0 ns=800000\nAFFINE bits=2 gs=128 rows=1 round=0 rep=15 arm=1 ns=500000\nAFFINE bits=3 gs=128 rows=1 round=0 rep=15 arm=0 ns=900000\nAFFINE bits=3 gs=128 rows=1 round=0 rep=15 arm=1 ns=600000'
        got = summarize(raw)
        self.assertEqual(got['affine_ms']['2/128']['1'], [0.8, 0.5])
        self.assertEqual(got['affine_ms']['3/128']['1'], [0.9, 0.6])


if __name__ == '__main__':
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(SummaryTest).run(result)
    if not result.wasSuccessful():
        for case, detail in result.errors + result.failures:
            print(detail, file=__import__('sys').stderr)
        raise SystemExit(1)

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ImageUp, Sparkles } from "lucide-react";
import { PredictionItem, predictImage } from "./api";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PredictionItem[]>([]);
  const [topK, setTopK] = useState(5);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const preds = await predictImage(file, topK);
      setResults(preds);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <div className="mx-auto max-w-5xl px-6 py-12">
        <motion.header
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-10"
        >
          <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-brand-400/30 bg-brand-500/10 px-4 py-1 text-sm text-brand-200">
            <Sparkles className="h-4 w-4" /> Recovered AI Model
          </p>
          <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">
            Premium Inference Console
          </h1>
          <p className="mt-3 max-w-2xl text-slate-300">
            Upload an image and run it through the reconstructed PyTorch checkpoint with fast API-backed predictions.
          </p>
        </motion.header>

        <div className="grid gap-6 md:grid-cols-2">
          <motion.form
            onSubmit={onSubmit}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-6 shadow-soft backdrop-blur"
          >
            <label className="mb-3 block text-sm font-medium text-slate-300">Image file</label>
            <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-600 bg-slate-800/60 p-8 transition hover:border-brand-300">
              <ImageUp className="h-8 w-8 text-brand-300" />
              <span className="text-sm text-slate-300">{file ? file.name : "Click to choose an image"}</span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>

            <div className="mt-5">
              <label className="mb-2 block text-sm font-medium text-slate-300">Top-K predictions</label>
              <input
                type="number"
                min={1}
                max={10}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 outline-none ring-brand-300 focus:ring"
              />
            </div>

            <button
              disabled={!file || loading}
              className="mt-6 w-full rounded-xl bg-brand-500 px-4 py-3 font-medium text-white transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Running inference..." : "Run Inference"}
            </button>

            {error && <p className="mt-4 text-sm text-red-300">{error}</p>}
          </motion.form>

          <motion.section
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-6"
          >
            <h2 className="mb-4 text-lg font-medium">Results</h2>
            {preview && (
              <img
                src={preview}
                alt="preview"
                className="mb-4 max-h-52 w-full rounded-xl object-cover"
              />
            )}
            <div className="space-y-2">
              {results.map((item, idx) => (
                <motion.div
                  key={`${item.index}-${idx}`}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="rounded-lg border border-slate-700 bg-slate-800/70 p-3"
                >
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{item.label}</span>
                    <span className="text-brand-300">{(item.score * 100).toFixed(2)}%</span>
                  </div>
                </motion.div>
              ))}
              {!results.length && <p className="text-sm text-slate-400">No predictions yet.</p>}
            </div>
          </motion.section>
        </div>
      </div>
    </div>
  );
}

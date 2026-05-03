export type PredictionItem = {
  index: number;
  label: string;
  score: number;
};

export async function predictImage(file: File, topK: number): Promise<PredictionItem[]> {
  const baseUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${baseUrl}/predict?top_k=${topK}`, {
    method: "POST",
    body: form
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Prediction failed with status ${response.status}`);
  }
  const payload = await response.json();
  return payload.predictions as PredictionItem[];
}

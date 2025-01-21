import Link from "next/link";

interface ModelListProps {
  models: {
    filename: string;
    content: any;
  }[];
  onEdit: (model: { filename: string; content: any }) => void;
}

export function ModelList({ models, onEdit }: ModelListProps) {
  return (
    <div className="border rounded-lg p-6 bg-white shadow-sm">
      <h2 className="text-xl font-semibold mb-4">Models</h2>
      <table className="min-w-full">
        <thead>
          <tr className="border-b">
            <th className="text-left py-2">Model Name</th>
            <th className="text-right py-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {models?.map((model) => (
            <tr key={model.filename} className="border-b">
              <td className="py-2">{model.filename}</td>
              <td className="text-right">
                <Link href={`models/${model.filename}/edit`}>Edit</Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

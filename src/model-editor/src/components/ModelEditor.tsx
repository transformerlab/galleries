'use client';

import { useState } from 'react';

interface ModelEditorProps {
  models: {
    filename: string;
    content: any;
  }[];
  onSave: () => void;
}

export function ModelEditor({ models, onSave }: ModelEditorProps) {
  const [editingModel, setEditingModel] = useState<string | null>(null);
  const [content, setContent] = useState('');
  const [error, setError] = useState('');

  const handleSave = async () => {
    try {
      JSON.parse(content);
      
      const response = await fetch(`/api/models/${editingModel}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: content,
      });

      if (!response.ok) {
        throw new Error('Failed to save');
      }

      setEditingModel(null);
      setError('');
      onSave();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Invalid JSON');
    }
  };

  if (editingModel) {
    const model = models.find(m => m.filename === editingModel);
    if (!model) return null;

    return (
      <div className="border rounded-lg p-6 bg-white shadow-sm">
        <h2 className="text-xl font-semibold mb-4">{model.filename}</h2>
        
        {error && (
          <div className="mb-4 p-4 bg-red-50 text-red-600 rounded">
            {error}
          </div>
        )}

        <div className="space-y-4">
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="w-full h-96 font-mono p-4 border rounded bg-gray-50"
          />
          <div className="space-x-2">
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Save
            </button>
            <button
              onClick={() => {
                setEditingModel(null);
                setContent('');
                setError('');
              }}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

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
          {/* {JSON.stringify(models)} */}
          {models?.map((model) => (
            <tr key={model.filename} className="border-b">
              <td className="py-2">{model.filename}</td>
              <td className="text-right">
                <button
                  onClick={() => {
                    setEditingModel(model.filename);
                    setContent(JSON.stringify(model.content, null, 2));
                  }}
                  className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Edit
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
} 
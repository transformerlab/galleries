'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ModelConfig } from '@/types/ModelConfig';

export default function EditModelPage() {
  const params = useParams();
  const router = useRouter();
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`/api/models/${params.filename}`);
        if (!response.ok) {
          throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        
        // Convert single model to array or use array directly
        const modelArray = Array.isArray(data) ? data : [data];
        setModels(modelArray.map(model => ({
          uniqueID: model.uniqueID || '',
          name: model.name || '',
          description: model.description || '',
          parameters: model.parameters || '',
          context: model.context || '',
          architecture: model.architecture || '',
          license: model.license || '',
        })));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load models');
      } finally {
        setIsLoading(false);
      }
    };

    fetchModels();
  }, [params.filename]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    try {
      const response = await fetch(`/api/models/${params.filename}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(models),
      });

      if (!response.ok) {
        throw new Error('Failed to update models');
      }

      router.push(`/models/${params.filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update models');
    }
  };

  const updateModel = (index: number, field: keyof ModelConfig, value: string) => {
    setModels(prevModels => {
      const newModels = [...prevModels];
      newModels[index] = { ...newModels[index], [field]: value };
      return newModels;
    });
  };

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (models.length === 0) return <div>No models found</div>;

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">
        <button 
          onClick={() => router.push(`/models/`)}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-gray-600 mr-4"
          >
            Back
        </button>
        Edit Model - {params.filename}
      </h1>
      
      <form onSubmit={handleSubmit} className="space-y-8">
        {models.map((model, index) => (
          <div key={index} className="border p-4 rounded-lg">
            <h2 className="text-xl font-semibold mb-4">Model {index + 1}</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Unique ID
                  <input
                    type="text"
                    value={model.uniqueID}
                    onChange={(e) => updateModel(index, 'uniqueID', e.target.value)}
                    className="w-full p-2 border rounded"
                  />
                </label>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Name
                  <input
                    type="text"
                    value={model.name}
                    onChange={(e) => updateModel(index, 'name', e.target.value)}
                    className="w-full p-2 border rounded"
                  />
                </label>
              </div>

              <div className="space-y-2 col-span-2">
                <label className="block text-sm font-medium text-gray-700">
                  Description
                  <textarea
                    value={model.description}
                    onChange={(e) => updateModel(index, 'description', e.target.value)}
                    className="w-full p-2 border rounded h-24"
                  />
                </label>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Parameters
                  <input
                    type="text"
                    value={model.parameters}
                    onChange={(e) => updateModel(index, 'parameters', e.target.value)}
                    className="w-full p-2 border rounded"
                  />
                </label>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Context
                  <input
                    type="text"
                    value={model.context}
                    onChange={(e) => updateModel(index, 'context', e.target.value)}
                    className="w-full p-2 border rounded"
                  />
                </label>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Architecture
                  <input
                    type="text"
                    value={model.architecture}
                    onChange={(e) => updateModel(index, 'architecture', e.target.value)}
                    className="w-full p-2 border rounded"
                  />
                </label>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  License
                  <input
                    type="text"
                    value={model.license}
                    onChange={(e) => updateModel(index, 'license', e.target.value)}
                    className="w-full p-2 border rounded"
                  />
                </label>
              </div>
            </div>
          </div>
        ))}

        <div className="flex gap-4 mt-6">
          <button
            type="submit"
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Save All Changes
          </button>
          <button
            type="button"
            onClick={() => router.push(`/models/`)}
            className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}

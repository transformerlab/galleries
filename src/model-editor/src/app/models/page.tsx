'use client';

import { useState, useEffect } from 'react';
import { ModelEditor } from '@/app/ui/ModelEditor';

interface Model {
  filename: string;
  content: any;
}

export default function Home() {
  const [models, setModels] = useState<Model[]>([]);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('/api/models');
      const data = await response.json();
      setModels(data);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  return (
    <main className="p-8 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Model Editor</h1>
      <div className="space-y-6">
          <ModelEditor 
            models={models}
            onSave={fetchModels}
          />
      </div>
    </main>
  );
} 
"use client";

import { useState } from "react";
import { ModelList } from "./ModelList";
import { ModelForm } from "./ModelForm";

interface ModelEditorProps {
  models: {
    filename: string;
    content: any;
  }[];
  onSave: () => void;
}

export function ModelEditor({ models, onSave }: ModelEditorProps) {
  const [editingModel, setEditingModel] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [error, setError] = useState("");

  const handleSave = async () => {
    try {
      JSON.parse(content);

      const response = await fetch(`/api/models/${editingModel}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: content,
      });

      if (!response.ok) {
        throw new Error("Failed to save");
      }

      setEditingModel(null);
      setError("");
      onSave();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid JSON");
    }
  };

  const handleCancel = () => {
    setEditingModel(null);
    setContent("");
    setError("");
  };

  const handleEdit = (model: { filename: string; content: any }) => {
    setEditingModel(model.filename);
    setContent(JSON.stringify(model.content, null, 2));
  };

  if (editingModel) {
    const model = models.find((m) => m.filename === editingModel);
    if (!model) return null;

    return (
      <ModelForm
        model={model}
        content={content}
        error={error}
        onContentChange={setContent}
        onSave={handleSave}
        onCancel={handleCancel}
      />
    );
  }

  return <ModelList models={models} onEdit={handleEdit} />;
}

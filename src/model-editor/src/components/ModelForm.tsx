interface ModelFormProps {
  model: {
    filename: string;
    content: any;
  };
  content: string;
  error: string;
  onContentChange: (content: string) => void;
  onSave: () => void;
  onCancel: () => void;
}

// Separate component for individual model form
function SingleModelForm({ 
  modelData, 
  index, 
  onChange 
}: { 
  modelData: any, 
  index?: number,
  onChange: (field: string, value: any, index?: number) => void 
}) {
  return (
    <div className="border rounded-lg p-6 bg-white shadow-sm mb-6">
      {index !== undefined && (
        <h3 className="text-lg font-medium mb-4">Model #{index + 1}</h3>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Unique ID
          </label>
          <input
            type="text"
            value={modelData.uniqueID || ''}
            onChange={(e) => onChange('uniqueID', e.target.value, index)}
            className="w-full p-2 border rounded"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Name
          </label>
          <input
            type="text"
            value={modelData.name || ''}
            onChange={(e) => onChange('name', e.target.value, index)}
            className="w-full p-2 border rounded"
          />
        </div>

        <div className="space-y-2 col-span-2">
          <label className="block text-sm font-medium text-gray-700">
            Description
          </label>
          <textarea
            value={modelData.description || ''}
            onChange={(e) => onChange('description', e.target.value, index)}
            className="w-full p-2 border rounded h-24"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Parameters
          </label>
          <input
            type="text"
            value={modelData.parameters || ''}
            onChange={(e) => onChange('parameters', e.target.value, index)}
            className="w-full p-2 border rounded"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Context
          </label>
          <input
            type="text"
            value={modelData.context || ''}
            onChange={(e) => onChange('context', e.target.value, index)}
            className="w-full p-2 border rounded"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Architecture
          </label>
          <input
            type="text"
            value={modelData.architecture || ''}
            onChange={(e) => onChange('architecture', e.target.value, index)}
            className="w-full p-2 border rounded"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            License
          </label>
          <input
            type="text"
            value={modelData.license || ''}
            onChange={(e) => onChange('license', e.target.value, index)}
            className="w-full p-2 border rounded"
          />
        </div>
      </div>
    </div>
  );
}

export function ModelForm({
  model,
  content,
  error,
  onContentChange,
  onSave,
  onCancel,
}: ModelFormProps) {
  // Parse the content to get the current values
  const modelData = JSON.parse(content);
  const isArray = Array.isArray(modelData);

  // Handle individual field changes
  const handleFieldChange = (field: string, value: any, index?: number) => {
    if (isArray) {
      const updatedContent = [...modelData];
      updatedContent[index!] = { ...updatedContent[index!], [field]: value };
      onContentChange(JSON.stringify(updatedContent, null, 2));
    } else {
      const updatedContent = { ...modelData, [field]: value };
      onContentChange(JSON.stringify(updatedContent, null, 2));
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold mb-4">{model.filename}</h2>
      
      {error && (
        <div className="mb-4 p-4 bg-red-50 text-red-600 rounded">
          {error}
        </div>
      )}

      <div className="space-y-4">
        {isArray ? (
          modelData.map((item: any, index: number) => (
            <SingleModelForm
              key={index}
              modelData={item}
              index={index}
              onChange={handleFieldChange}
            />
          ))
        ) : (
          <SingleModelForm
            modelData={modelData}
            onChange={handleFieldChange}
          />
        )}

        <div className="mt-8 space-x-2">
          <button
            onClick={onSave}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Save
          </button>
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
} 
interface PromptInputProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  drawMode: boolean;
  editMode: boolean;
  onReset: () => void;
}

export const PromptInput = ({ 
  prompt, 
  onPromptChange, 
  onSubmit, 
  isLoading, 
  drawMode,
  editMode,
  onReset
}: PromptInputProps) => (
  <div className="flex gap-4 w-full max-w-4xl mx-auto">
    <div className="form-control flex-grow">
      <input
        type="text"
        value={prompt}
        onChange={(e) => onPromptChange(e.target.value)}
        placeholder={
          editMode 
            ? "Type your message..." 
            : drawMode 
              ? "E.g., 'A sunset over mountains'" 
              : "E.g., 'Abstract art with circles'"
        }
        className="input input-bordered w-full"
        disabled={isLoading}
      />
      <label className="label">
        <span className="label-text-alt text-gray-500">
          {editMode 
            ? "Ask to make edits to your drawing" 
            : drawMode 
              ? "Be specific with your description for better results" 
              : "Enter keywords to find similar artwork"}
        </span>
      </label>
    </div>
    <div className="flex gap-2">
      {editMode && (
        <button
          onClick={onReset}
          className="btn btn-warning"
          disabled={isLoading}
          title="Reset conversation and clear image"
        >
          Reset
        </button>
      )}
      <button 
        onClick={onSubmit}
        className={`btn ${editMode ? 'btn-info' : drawMode ? 'btn-success' : 'btn-primary'} min-w-[120px]`}
        disabled={isLoading || !prompt.trim()}
        title={!prompt.trim() ? "Please enter a message first" : ""}
      >
        {isLoading ? (
          <>
            <span className="loading loading-spinner"></span>
            {editMode ? "Editing..." : drawMode ? "Generating..." : "Finding..."}
          </>
        ) : (editMode ? "Edit" : drawMode ? "Generate Code" : "Find")}
      </button>
    </div>
  </div>
); 
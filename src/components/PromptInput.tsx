interface PromptInputProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  drawMode: boolean;
}

export const PromptInput = ({ 
  prompt, 
  onPromptChange, 
  onSubmit, 
  isLoading, 
  drawMode 
}: PromptInputProps) => (
  <div className="flex gap-4 w-full max-w-4xl mx-auto">
    <div className="form-control flex-grow">
      <input
        type="text"
        value={prompt}
        onChange={(e) => onPromptChange(e.target.value)}
        placeholder={drawMode ? "E.g., 'A sunset over mountains'" : "E.g., 'Abstract art with circles'"}
        className="input input-bordered w-full"
        disabled={isLoading}
      />
      <label className="label">
        <span className="label-text-alt text-gray-500">
          {drawMode ? "Be specific with your description for better results" : "Enter keywords to find similar artwork"}
        </span>
      </label>
    </div>
    <button 
      onClick={onSubmit}
      className={`btn ${drawMode ? 'btn-success' : 'btn-primary'} min-w-[120px]`}
      disabled={isLoading || !prompt.trim()}
      title={!prompt.trim() ? "Please enter a description first" : ""}
    >
      {isLoading ? (
        <>
          <span className="loading loading-spinner"></span>
          {drawMode ? "Generating..." : "Finding..."}
        </>
      ) : (drawMode ? "Generate Code" : "Find")}
    </button>
  </div>
); 
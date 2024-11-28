interface PromptInputProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  placeholder: string;
  helperText: string;
  submitButtonText: string;
  loadingText: string;
  showResetButton?: boolean;
  onReset?: () => void;
  submitButtonClass?: string;
}

export const PromptInput = ({ 
  prompt, 
  onPromptChange, 
  onSubmit, 
  isLoading,
  placeholder,
  helperText,
  submitButtonText,
  loadingText,
  showResetButton = false,
  onReset,
  submitButtonClass = 'btn-primary'
}: PromptInputProps) => (
  <div className="flex gap-4 w-full max-w-4xl mx-auto">
    <div className="form-control flex-grow">
      <input
        type="text"
        value={prompt}
        onChange={(e) => onPromptChange(e.target.value)}
        placeholder={placeholder}
        className="input input-bordered w-full"
        disabled={isLoading}
      />
      <label className="label">
        <span className="label-text-alt text-gray-500">{helperText}</span>
      </label>
    </div>
    <div className="flex gap-2">
      <button 
        onClick={onSubmit}
        className={`btn ${submitButtonClass} min-w-[120px]`}
        disabled={isLoading || !prompt.trim()}
        title={!prompt.trim() ? "Please enter a message first" : ""}
      >
        {isLoading ? (
          <>
            <span className="loading loading-spinner"></span>
            {loadingText}
          </>
        ) : submitButtonText}
      </button>
      {showResetButton && onReset && (
        <button
          onClick={onReset}
          className="btn btn-warning"
          disabled={isLoading}
          title="Reset conversation and clear image"
        >
          Reset
        </button>
      )}
    </div>
  </div>
); 
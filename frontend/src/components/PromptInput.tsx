import { useEffect, useState } from 'react';
import { useAnalytics } from 'rashik-analytics-provider';

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
}: PromptInputProps) => {
  const { trackEvent } = useAnalytics();
  const [lastPrompt, setLastPrompt] = useState('');

  // Track when user submits a prompt
  const handleSubmit = () => {
    trackEvent('prompt_submitted', { 
      prompt_length: prompt.length,
      prompt_text: prompt,
      button_type: submitButtonText 
    });
    onSubmit();
  };

  // Track when user resets
  const handleReset = () => {
    if (onReset) {
      trackEvent('prompt_reset');
      onReset();
    }
  };

  // Track prompt changes after a delay
  useEffect(() => {
    if (prompt !== lastPrompt && prompt.trim()) {
      const timeoutId = setTimeout(() => {
        trackEvent('prompt_typing', { prompt_length: prompt.length });
        setLastPrompt(prompt);
      }, 2000); // Only track after 2 seconds of inactivity
      
      return () => clearTimeout(timeoutId);
    }
  }, [prompt, trackEvent, lastPrompt]);

  return (
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
          onClick={handleSubmit}
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
            onClick={handleReset}
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
} 
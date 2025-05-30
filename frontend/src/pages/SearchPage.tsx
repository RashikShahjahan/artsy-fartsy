import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { findSimilarArt } from '../api';
import { PromptInput } from '../components/PromptInput';
import { SimilarDrawings } from '../components/SimilarDrawings';
import { Alert } from '../components/Alert';
import { useAnalytics } from 'rashik-analytics-provider';

function SearchPage() {
  const [prompt, setPrompt] = useState('');
  const [similarDrawings, setSimilarDrawings] = useState<string[]>([]);
  const [isFinding, setIsFinding] = useState(false);
  const [alert, setAlert] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  
  const navigate = useNavigate();
  const { trackEvent } = useAnalytics();

  const findSimilar = async () => {
    setIsFinding(true);
    trackEvent('search_drawings', { search_prompt: prompt });
    
    try {
      const drawings = await findSimilarArt(prompt,'drawing');
      setSimilarDrawings(drawings);
      trackEvent('search_success', { 
        search_prompt: prompt,
        results_count: drawings.length
      });
    } catch (error) {
      trackEvent('search_error', {
        search_prompt: prompt,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      
      setAlert({ message: 'Error finding similar drawings', type: 'error' });
    } finally {
      setIsFinding(false);
    }
  };

  const navigateToDrawing = () => {
    trackEvent('navigate_to_drawing');
    navigate('/');
  };

  return (
    <>
      {alert && (
        <Alert
          message={alert.message}
          type={alert.type}
          onClose={() => {
            trackEvent('close_alert', { alert_type: alert.type });
            setAlert(null);
          }}
        />
      )}
      
      <PromptInput 
        prompt={prompt}
        onPromptChange={setPrompt}
        onSubmit={findSimilar}
        isLoading={isFinding}
        placeholder="E.g., 'Abstract art with circles'"
        helperText="Enter keywords to find similar artwork"
        submitButtonText="Find"
        loadingText="Finding..."
        submitButtonClass="btn-primary"
      />

      <SimilarDrawings 
        drawings={similarDrawings}
        isFinding={isFinding}
      />

      <button 
        onClick={navigateToDrawing}
        className="btn btn-secondary w-full max-w-xs mx-auto"
      >
        Back to Drawing
      </button>
    </>
  );
}

export default SearchPage; 
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { findSimilarArt } from '../api';
import { PromptInput } from '../components/PromptInput';
import { SimilarDrawings } from '../components/SimilarDrawings';
import { Alert } from '../components/Alert';

function SearchPage() {
  const [prompt, setPrompt] = useState('');
  const [similarDrawings, setSimilarDrawings] = useState<string[]>([]);
  const [isFinding, setIsFinding] = useState(false);
  const [alert, setAlert] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  
  const navigate = useNavigate();

  const findSimilar = async () => {
    setIsFinding(true);
    try {
      const drawings = await findSimilarArt(prompt,'drawing');
      setSimilarDrawings(drawings);
    } catch (error) {
      setAlert({ message: 'Error finding similar drawings', type: 'error' });
    } finally {
      setIsFinding(false);
    }
  };

  return (
    <>
      {alert && (
        <Alert
          message={alert.message}
          type={alert.type}
          onClose={() => setAlert(null)}
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
        onClick={() => navigate('/')}
        className="btn btn-secondary w-full max-w-xs mx-auto"
      >
        Back to Drawing
      </button>
    </>
  );
}

export default SearchPage; 
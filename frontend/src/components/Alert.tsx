import { useEffect, useState } from 'react';

interface AlertProps {
  message: string;
  type?: 'success' | 'error';
  duration?: number;
  onClose?: () => void;
}

export function Alert({ message, type = 'error', duration = 3000, onClose }: AlertProps) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      onClose?.();
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const baseStyles = "fixed top-4 right-4 p-4 rounded-lg shadow-lg transform transition-all duration-300";
  const typeStyles = type === 'success' 
    ? "bg-green-100 text-green-800 border border-green-200"
    : "bg-red-100 text-red-800 border border-red-200";
  const visibilityStyles = isVisible
    ? "translate-x-0 opacity-100"
    : "translate-x-full opacity-0";

  return (
    <div className={`${baseStyles} ${typeStyles} ${visibilityStyles}`}>
      {message}
    </div>
  );
} 
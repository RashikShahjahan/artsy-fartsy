import React from 'react';
import { SignInButton } from '@clerk/clerk-react';

const SignInButtonWrapper: React.FC = () => {
  return (
    <div className="absolute bottom-6 left-6 right-6 px-6 py-3 bg-yellow-400 text-black hover:bg-yellow-350 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50 text-center">
      <SignInButton>
        Sign in to save your art!
      </SignInButton>
    </div>
  );
};

export default SignInButtonWrapper;

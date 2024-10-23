import React from 'react';
import { SignInButton } from '@clerk/clerk-react';

const SignInButtonWrapper: React.FC<{text: string}> = ({text}) => {
  return (
      <SignInButton>
        <button className="absolute bottom-6 left-6 right-6 px-6 py-3 bg-yellow-400 text-black hover:bg-yellow-350 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50 text-center">
                {text}
        </button>
      </SignInButton>
  );
};

export default SignInButtonWrapper;

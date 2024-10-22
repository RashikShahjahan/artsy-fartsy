import { SignedIn, SignedOut, SignInButton, SignOutButton } from '@clerk/clerk-react'
import DrawingBoard from './DrawingBoard'


/*Tasks:
Saving Drawings and viewing gallery(Add auth)
  - Create button to save drawings
  - Axios POST request to /save_art
  - View to Display drawings in their feed in a news feed(Make it like a gallery)
  - GET request to retrieve_gallery
  - Post request to like/unlike drawings

Enhancements:
  - User profile page
  - View to display user's saved drawings
  - View to display user's followers and following
  - Help button to help artists with code syntax
  - Call LLM to generate code
*/


export default function App() {
  return (
    <header>
      <SignedOut>
        <div>
          <SignInButton />
          <DrawingBoard />
        </div>
      </SignedOut>
      <SignedIn>
        <div>
          <DrawingBoard />
          <SignOutButton />
        </div>
      </SignedIn>
    </header>
  )
}

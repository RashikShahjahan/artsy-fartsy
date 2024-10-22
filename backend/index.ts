/*Tasks:
Authentication
  - Clerk middleware to protect routes
  - Add clerk auth logic to the endpoints

Database
  - Setup Prisma and connect to PostgreSQL
  - Create prisma schema to store images and user 
 
Endpoints
  - Create endpoint to manage login, signup, and logout
  - POST endpoint save_art
  - GET endpoint retrieve_gallery
  - POST endpoint like_art
  - POST endpoint unlike_art
  - endpoint to make LLM call
  - GET endpoint retrieve_user_arts(Optional)
  - GET endpoint retrieve_user_followers(Optional)
  - GET endpoint retrieve_user_following(Optional)
  - POST endpoint follow_user(Optional)
  - POST endpoint unfollow_user(Optional)   

Deployment:
  - Dockerize the app(Optional)
  - Deploy the app 
*/

import express from 'express';
import cors from 'cors';
import { interpret } from './interpret';

const app = express();
app.use(cors());
app.use(express.json());

app.post('/interpret', (req, res) => {
    console.log(req.body);
    const { code } = req.body;
    const commands = interpret(code);
    res.json(commands);
});

app.listen(3001, () => {
    console.log('Server is running on port 3001');
});
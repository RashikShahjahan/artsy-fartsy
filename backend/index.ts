/*Tasks:

Database
  - Setup Prisma and connect to PostgreSQL
  - Create prisma schema to store images and user 

Art Gallery
  - POST endpoint save_art
  - GET endpoint retrieve_gallery
  - POST endpoint like_art
  - POST endpoint unlike_art

Enhancements
  - GET endpoint retrieve_user_arts(Optional)
  - GET endpoint retrieve_user_followers(Optional)
  - GET endpoint retrieve_user_following(Optional)
  - POST endpoint follow_user(Optional)
  - POST endpoint unfollow_user(Optional)   

Deployment:
  - Dockerize the app(Optional)
  - Deploy the app 
*/
import express, {type NextFunction, type Request, type Response} from 'express';
import { interpret } from './interpret';
import 'dotenv/config';
import cors from 'cors';
import { clerkClient, clerkMiddleware, getAuth} from '@clerk/express'

async function identifyUserMiddleware(req: Request, res: Response, next: NextFunction) {
    const { userId } = getAuth(req);
    console.log(userId);

    next();
}

const app = express();
app.use(cors());
app.use(express.json());
app.use(clerkMiddleware());
app.use(identifyUserMiddleware);

app.post('/interpret', async(req, res) => {
    const { code } = req.body;
    const commands = interpret(code);
    res.json(commands);
});

app.listen(3001, () => {
    console.log('Server is running on port 3001');
});
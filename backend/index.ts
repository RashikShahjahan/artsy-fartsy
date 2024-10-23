import express, {type NextFunction, type Request, type Response} from 'express';
import { interpret, type Command } from './interpret';
import 'dotenv/config';
import cors from 'cors';
import { clerkClient, clerkMiddleware, getAuth} from '@clerk/express'


type SavedArt = {
  id: number;
  username: string;
  likes: number;
  commands: Command[];
};


const app = express();
app.use(cors());
app.use(express.json());
app.use(clerkMiddleware());

app.post('/interpret', async(req, res) => {
    const { code } = req.body;
    const commands = interpret(code);
    res.json(commands);
});

app.post('/save_art', async (req, res) => {
  const { drawCommands } = req.body;
  const { userId } = getAuth(req);
  if (!userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  

  const newArt: SavedArt = {
    id: 0,
    username: userId,
    likes: 0,
    commands: drawCommands
  };


  res.json({message: 'Art saved', id: newArt.id});
});

app.get('/get_art/:artId', async (req, res) => {
  const { userId } = getAuth(req);
  if (!userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }


});

app.get('/get_previous_art', async (req, res) => {
  const { userId } = getAuth(req);
  if (!userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  
});

app.get('/get_next_art', async (req, res) => {
  const { userId } = getAuth(req);
  if (!userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }


});

app.listen(3001, () => {
    console.log('Server is running on port 3001');
});

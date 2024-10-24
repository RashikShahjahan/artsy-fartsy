import express, {type NextFunction, type Request, type Response} from 'express';
import { interpret, type Command } from './interpret';
import 'dotenv/config';
import cors from 'cors';
import { clerkClient, clerkMiddleware, getAuth} from '@clerk/express';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();


const app = express();

// Configure CORS
const allowedOrigin = process.env.ALLOWED_ORIGIN;
app.use(cors({
  origin: allowedOrigin,
  credentials: true,
}));

app.use(express.json());
app.use(clerkMiddleware());
app.use(identifyUserMiddleware);

async function identifyUserMiddleware(req: Request, res: Response, next: NextFunction) {
    const { userId } = getAuth(req);

    const clerkUser = await clerkClient.users.getUser(userId);
    let user = await prisma.user.findUnique({
        where:{
            clerkId:userId
        }
    });

    if (!user){
        user = await prisma.user.create({
            data: {
                username: clerkUser.username ?? userId,
                clerkId: userId
            }
        });
    };

    req.user = user;
    next();

};

app.post('/interpret', async(req, res) => {
    const { code } = req.body;
    const commands = interpret(code);
    res.json(commands);
});

app.post('/save_art', async (req: Request, res: Response) => {
  const { drawCommands } = req.body;
  if (!req.user) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const art = await prisma.art.create({
    data: {
      userId: req.user.id,
      commands: drawCommands
    }
  });

  res.json({artId: art.id});
});

app.get('/get_art', async (req, res) => {
  const { skip } = req.query;

  const art = await prisma.art.findMany({
    skip: parseInt(skip as string),
    take: 1
  });
  res.json({artData: art});
});

app.get('/get_art_count', async (req, res) => {
  const artCount = await prisma.art.count();
  res.json({artCount});
});

app.post('/like_art', async (req, res) => {
  const { artId } = req.body;
  await prisma.art.update({
    where: { id: artId },
    data: { likes: { increment: 1 } }
  });
  await prisma.user.update({
    where: { id: req.user.id },
    data: { likedArts: { connect: { id: artId } } }
  });
});

app.post('/unlike_art', async (req, res) => {
  const { artId } = req.body;
  await prisma.art.update({
    where: { id: artId },
    data: { likes: { decrement: 1 } }
  });
  await prisma.user.update({
    where: { id: req.user.id },
    data: { likedArts: { disconnect: { id: artId } } }
  });
});

app.get('/get_liked_status', async (req, res) => {
  const { artId } = req.query;
  const likedStatus = await prisma.user.findFirst({ where: { id: req.user.id, likedArts: { some: { id: artId } } } });
  res.json({ isLiked: likedStatus !== null});
});

app.get('/get_likes', async (req, res) => {
  const { artId } = req.query;
  const likes = await prisma.art.findUnique({ where: { id: artId } });
  res.json({likes: likes?.likes});
});

app.listen(3001, () => {
    console.log('Server is running on port 3001');
});

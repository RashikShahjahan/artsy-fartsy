import express, {type NextFunction, type Request, type Response} from 'express';
import { interpret, type Command } from './interpret';
import 'dotenv/config';
import cors from 'cors';
import { clerkClient, clerkMiddleware, getAuth} from '@clerk/express';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();


const app = express();
app.use(cors());
app.use(express.json());
app.use(clerkMiddleware());
app.use(identifyUserMiddleware);



async function identifyUserMiddleware(req: Request, res: Response, next: NextFunction) {
    const { userId } = getAuth(req);

    if (userId) {
        const clerkUser = await clerkClient.users.getUser(userId);
        let user = await prisma.user.findUnique({
            where: {
                clerkId: userId
            }
        });

        if (!user) {
            user = await prisma.user.create({
                data: {
                    username: clerkUser.username ?? userId,
                    clerkId: userId
                }
            });
        }

        req.user = user;
        next();
    } else {
        next();
    }
}

app.post('/interpret', async (req: Request, res: Response) => {
    const { code } = req.body;
    const commands = interpret(code);
    res.json(commands);
});

  app.post('/save_art', async (req: Request, res: Response) => {
  const { drawCommands } = req.body;
  const user = req.user;
  if (!user) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const art = await prisma.art.create({
    data: {
      userId: user.id,
      commands: drawCommands
    }
  });

  res.json({artId: art.id});
});

app.get('/get_art', async (req: Request, res: Response) => {
  const { skip } = req.query;
  const skipValue = typeof skip === 'string' ? parseInt(skip, 10) : 0;

  const art = await prisma.art.findMany({
    skip: skipValue,
    take: 1
  });
  res.json({artData: art});
});

app.get('/get_art_count', async (req: Request, res: Response) => {
  const artCount = await prisma.art.count();
  res.json({artCount});
});

app.post('/like_art', async (req: Request, res: Response) => {
  const { artId } = req.body;
  if (typeof artId !== 'string') {
    return res.status(400).json({ error: 'Invalid artId' });
  }
  await prisma.art.update({
    where: { id: artId },
    data: { likes: { increment: 1 } }
  });
  if (req.user) {
    await prisma.user.update({
      where: { id: req.user.id },
      data: { likedArts: { connect: { id: artId } } }
    });
  }
  res.json({ success: true });
});

app.post('/unlike_art', async (req: Request, res: Response) => {
  const { artId } = req.body;
  if (typeof artId !== 'string') {
    return res.status(400).json({ error: 'Invalid artId' });
  }
  await prisma.art.update({
    where: { id: artId },
    data: { likes: { decrement: 1 } }
  });
  if (req.user) {
    await prisma.user.update({
      where: { id: req.user.id },
      data: { likedArts: { disconnect: { id: artId } } }
    });
  }
  res.json({ success: true });
});

  app.get('/get_liked_status', async (req: Request, res: Response) => {
  const { artId } = req.query;
  if (!req.user) {
    return res.status(401).json({ error: 'User not authenticated' });
  }
  if (typeof artId !== 'string') {
    return res.status(400).json({ error: 'Invalid artId' });
  }
  const likedStatus = await prisma.user.findFirst({
    where: {
      id: req.user.id,
      likedArts: { some: { id: artId } }
    }
  });
  res.json({ isLiked: likedStatus !== null });
});

app.get('/get_likes', async (req: Request, res: Response) => {
  const { artId } = req.query;
  if (typeof artId !== 'string') {
    return res.status(400).json({ error: 'Invalid artId' });
  }
  const art = await prisma.art.findUnique({ where: { id: artId } });
  res.json({ likes: art?.likes ?? 0 });
});

app.listen(3001, () => {
    console.log('Server is running on port 3001');
});

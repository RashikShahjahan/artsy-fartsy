import express, {type NextFunction, type Request, type Response} from 'express';
import { interpret} from './interpret';
import 'dotenv/config';
import cors from 'cors';
import { clerkClient, clerkMiddleware, getAuth} from '@clerk/express';
import { PrismaClient } from '@prisma/client';
import { generateCode } from './llm';

const prisma = new PrismaClient();


const app = express();
app.use(cors({
  origin: process.env.ALLOWED_ORIGIN,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Authorization', 'Content-Type'],
  credentials: true
}));
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
  const skipValue = parseInt(skip as string);

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

app.post('/toggle_like', async (req: Request, res: Response) => {
  const { artId } = req.body;
  const userId = req.user?.id;

  if (!userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  // Find existing like
  const existingLike = await prisma.like.findFirst({
    where: {
      userId: userId,
      artId: artId
    }
  });

  if (existingLike) {
    // User already liked, so remove like
    await prisma.like.delete({
      where: {
        id: existingLike.id
      }
    });

    // Decrement likes count, ensuring it doesn't go below 0
    await prisma.art.update({
      where: { id: artId },
      data: {
        likes: {
          decrement: 1
        }
      }
    });
  } else {
    // User hasn't liked, so add like
    await prisma.like.create({
      data: {
        userId: userId,
        artId: artId
      }
    });

    // Increment likes count
    await prisma.art.update({
      where: { id: artId },
      data: {
        likes: {
          increment: 1
        }
      }
    });
  }
  res.json({ success: true, isLiked: !existingLike});
});
  
app.get('/is_liked', async (req: Request, res: Response) => {
  const { artId } = req.query;
  const userId = req.user?.id;
  const isLiked = await prisma.like.findFirst({ where: { userId, artId }});
  res.json({ isLiked: !!isLiked});
});


app.get('/get_likes', async (req: Request, res: Response) => {
  const { artId } = req.query;
  const art = await prisma.art.findUnique({ where: { id: artId as string  } });
  res.json({ likes: art?.likes ?? 0 });
});

app.post('/generate_code', async (req: Request, res: Response) => {
  const { userPrompt } = req.body;
  const code = await generateCode(userPrompt);
  const commands = interpret(code);
  res.json(commands);
});




app.listen(3000,"0.0.0.0", () => {
    console.log('Server is running on port 3000');
});

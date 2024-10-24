-- CreateTable
CREATE TABLE "Art" (
    "id" SERIAL NOT NULL,
    "userId" TEXT NOT NULL,
    "likes" INTEGER NOT NULL DEFAULT 0,
    "commands" JSONB NOT NULL,

    CONSTRAINT "Art_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "username" TEXT NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "Art" ADD CONSTRAINT "Art_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

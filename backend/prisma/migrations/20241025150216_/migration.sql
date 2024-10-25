/*
  Warnings:

  - You are about to drop the `_UserLikes` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "_UserLikes" DROP CONSTRAINT "_UserLikes_A_fkey";

-- DropForeignKey
ALTER TABLE "_UserLikes" DROP CONSTRAINT "_UserLikes_B_fkey";

-- DropTable
DROP TABLE "_UserLikes";

-- CreateTable
CREATE TABLE "Like" (
    "id" TEXT NOT NULL,
    "artId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,

    CONSTRAINT "Like_pkey" PRIMARY KEY ("id")
);

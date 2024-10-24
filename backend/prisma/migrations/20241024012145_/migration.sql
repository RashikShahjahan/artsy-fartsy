/*
  Warnings:

  - The primary key for the `Art` table will be changed. If it partially fails, the table could be left without primary key constraint.

*/
-- AlterTable
ALTER TABLE "Art" DROP CONSTRAINT "Art_pkey",
ALTER COLUMN "id" DROP DEFAULT,
ALTER COLUMN "id" SET DATA TYPE TEXT,
ADD CONSTRAINT "Art_pkey" PRIMARY KEY ("id");
DROP SEQUENCE "Art_id_seq";

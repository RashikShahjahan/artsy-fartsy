-- CreateTable
CREATE TABLE "_UserLikes" (
    "A" TEXT NOT NULL,
    "B" TEXT NOT NULL
);

-- CreateIndex
CREATE UNIQUE INDEX "_UserLikes_AB_unique" ON "_UserLikes"("A", "B");

-- CreateIndex
CREATE INDEX "_UserLikes_B_index" ON "_UserLikes"("B");

-- AddForeignKey
ALTER TABLE "_UserLikes" ADD CONSTRAINT "_UserLikes_A_fkey" FOREIGN KEY ("A") REFERENCES "Art"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_UserLikes" ADD CONSTRAINT "_UserLikes_B_fkey" FOREIGN KEY ("B") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

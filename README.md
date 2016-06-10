READ BEFORE CLONING
----------------------------------------------------------------
**This repo is enormous, several GB big. I decided to save all my models from all my
experiments, some of which took up quite a bit of space because I didn't even try
for space efficiency. Cloning this will take a long time and a lot of space. I'm
not saying you can't do it, just know what you're getting into.**

This is the code I'm using for a research side project. Basically, I'm throwing
neural nets at a Rubik's Cube to see how well the neural net can learn
the dynamics of a Rubik's Cube problem. Past research has shown the Rubik's
Cube is solvable with iteratively deepened A\* search, so this isn't about
doing something believed to be difficult. I'm more interested in finding out
how well a neural net can recreate/imitate search.

If I have time I'll update this README. This is highly unrefined research code
and as of now I have no plans to clean it up or make
it more readable to other people.
Off the top of my head, you'll need Torch and the rnn library for Torch. (There's
code for parallel data set generation that needs the parallel library, but turns out
dataset generation is fast enough to make it a premature optimization, and you shouldn't
need it for training.)

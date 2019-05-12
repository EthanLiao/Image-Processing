#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <queue>
using namespace cv;
using namespace std;

struct Point_element
{
  int row,col;
  int dist,dir; //distance for the queue node
};


struct deviate
{
  int vert,horiz;
};

deviate move_short[4]={{0,1},{1,0},{0,-1},{-1,0}};
                    /* right  dowwn left   upper*/

int **maze_short,**mark_short;
int **maze_short_final; //For the path distacne record

class BFS_Algo_for_graph
{
private :
 int row,column;              // index for the whole maze
 int start_row,start_column; //index for the start position
 int end_row,end_column;     //index for the exist position
 int nRows,nCols;
 Mat image;
public:
	~BFS_Algo_for_graph();
	void display();
  void readBmpFile(string);
	void solveMaze(void);	
	void traceBack(void);
};

BFS_Algo_for_graph :: ~BFS_Algo_for_graph()
{
  delete [] maze_short;
  delete [] maze_short_final;
  delete [] mark_short;
}

/*Read the image and then construct a image map*/
void BFS_Algo_for_graph :: readBmpFile(string srcFilename)
{
   image=imread(srcFilename,CV_LOAD_IMAGE_GRAYSCALE);
   if(!image.data) {cout<<"coud't open or find the image"<<"\n"; exit(EXIT_FAILURE);}
   
   //Sum up the datastructure of the graph
   //int channels=image.channels();
   nRows=image.rows; nCols=image.cols; /* For n channel nCols=image.cols*channels;*/
   start_row=0;start_column=0;end_row=nRows;end_column=nCols;
   
   //if(image.isContinuous()){ nCols *=nRows; nRows=1;}
   
   // Build up the integer map for the DFS Algo
   maze_short=new int*[nRows];
   mark_short=new int*[nRows];
   maze_short_final=new int*[nRows];
   for(int k=0;k<nRows;k++)
    {
      maze_short[k]=new int[nCols];
      mark_short[k]=new int[nCols];
      maze_short_final[k]=new int[nCols];
    }
   
   // access all the pixel single channel in the graph
   for(int i=0;i<nRows;i++)
       for(int j=0;j<nCols;j++)
       {
         Vec3b intensity=image.at<Vec3b>(i,j); 
         maze_short[i][j]=intensity.val[0];
       	 mark_short[i][j]=maze_short[i][j];
         maze_short_final[i][j]=-1;
       } 
}

void BFS_Algo_for_graph :: solveMaze()
{
 int row,col,dir;
 int nextRow,nextCol;
 int found=0;
 
 // maze entrance & distance
 Point_element position_1,nextPosition;
 position_1.row=start_row;
 position_1.col=start_column;
 position_1.dist=0; // intial the distance of start position_1 is 0
 position_1.dir=0;
 
 maze_short_final[start_row][start_column]=position_1.dist;

 
 // initialize queue
 queue<Point_element> q;
 q.push(position_1);
 while(!q.empty())
 {
 	position_1=q.front();
 	row=position_1.row;
 	col=position_1.col;
  dir=position_1.dir;
  q.pop();
  
  if(found==1) break; // found the exit
 	
 	while(dir<4) // if there exist more direction to go
 	 {
 	  //cordinate of the next move
 	  nextRow=row+move_short[dir].vert;
 	  nextCol=col+move_short[dir].horiz;

    //if the next postion don't encounter with the wall and havn't been visted
    //it imply that it's a true direction 
    if((mark_short[nextRow][nextCol]!=-1) && (mark_short[nextRow][nextCol]==20 ||
        mark_short[nextRow][nextCol]==85||mark_short[nextRow][nextCol]==150||
        mark_short[nextRow][nextCol]==255))
 	 	{
 	 	  mark_short[nextRow][nextCol]=-1; // mark the next position to indecate that we have traversed
 	 	  // enqueue the current position_1
      // we will set the default direction to try for the next position */
      nextPosition.row=nextRow;
      nextPosition.col=nextCol;
      nextPosition.dist=position_1.dist+1;
      maze_short_final[nextRow][nextCol]=nextPosition.dist;
      nextPosition.dir=0;
      q.push(nextPosition);
      dir++;
      if((nextRow==end_row) && (nextCol==end_column)){found=1;break;}
 	 	}
 	 	//if the next position_1 is the wall or have been here
 	 	else ++dir;
 	 }
 }
}

deviate trace_back[4]={{0,-1},{-1,0},{0,1},{1,0},};
                      /* left  upper  right  down */
void BFS_Algo_for_graph :: traceBack()
{
   int row,col,dir,dist;
   int nextRow,nextCol;
   // maze entrance & distance
   Point_element position;
   position.row=end_row;
   position.col=end_column;
   position.dir=0;
   position.dist=maze_short_final[end_row][end_column];

   //cordinate and direcrion pop from stack
   row=position.row;
   col=position.col;
   dist=position.dist;
   dir=position.dir;
   while(dir<4) // if there exist more direction to go
   {
    //cordinate of the next move
    nextRow=row+trace_back[dir].vert;
    nextCol=col+trace_back[dir].horiz;
    // if the next positon is the exit
    if((nextRow==start_row) && (nextCol==start_column))
    {
      // mark the next position
      if(maze_short[nextRow][nextCol]==20)maze_short[nextRow][nextCol]=-1;
      if(maze_short[nextRow][nextCol]==85)maze_short[nextRow][nextCol]=-2;
      if(maze_short[nextRow][nextCol]==150)maze_short[nextRow][nextCol]=-3;
      if(maze_short[nextRow][nextCol]==255)maze_short[nextRow][nextCol]=-4;
      break;
    }
    //if the next postion don't encounter with the wall and havn't been visted
    else if
    (
      ((nextRow!=row-1 || nextRow!=0 || nextCol!=column-1 || nextCol!=0))
        && maze_short_final[nextRow][nextCol]==dist-1
    )
    {
      // mark the next position
      if(maze_short[nextRow][nextCol]==20)maze_short[nextRow][nextCol]=-1;
      if(maze_short[nextRow][nextCol]==85)maze_short[nextRow][nextCol]=-2;
      if(maze_short[nextRow][nextCol]==150)maze_short[nextRow][nextCol]=-3;
      if(maze_short[nextRow][nextCol]==255)maze_short[nextRow][nextCol]=-4;
      row=nextRow; col=nextCol; dir=0; dist--; // For the next position,we set default direction to try 
    }
    //if the next position is the wall or have been here
    else ++dir;
  }
}

void BFS_Algo_for_graph :: display()
{
  traceBack();
  for (int i=0;i<nRows;i++)
   for(int j=0;j<nCols;j++)
   {
      Vec3b &intensity=image.at<Vec3b>(i,j);
      intensity.val[0]=maze_short[i][j];
   }

  namedWindow("Display window",WINDOW_AUTOSIZE);
  imshow("Display window",image);
  waitKey(0);
}

int main(int argc, char const *argv[])
{
   cout<<"test";
   BFS_Algo_for_graph obj_2;
   obj_2.readBmpFile(argv[1]);
   obj_2.solveMaze();
   obj_2.display();

	return 0;
}

#include "MultiCutTrack.h"

int main(int argc, char** argv)
{
  char* vertex_file = argv[1];
  char* edge_file = argv[2];
  char* track_file = argv[3];
  MulticutTrack multicut_track;
  multicut_track.LoadTrackDat(vertex_file, edge_file);
  multicut_track.MulticutProcess();
  multicut_track.DumpToFile(track_file);
  multicut_track.LoadTrack(track_file);
  return 0;
}

#include <andres/graph/graph.hxx>
#include "andres/graph/multicut/kernighan-lin.hxx"
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
const int k_track_len_thresh = 5;
inline bool file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    bool flag = f.good();
    f.close();
    return flag;
}

struct EdgeCost
{
    EdgeCost(){}
    EdgeCost(int s, int e, float c):start_vertical(s), end_vertical(e), edge_cost(c){}
    int start_vertical;
    int end_vertical;
    float edge_cost;
};

struct TrackBoxRead
{
  int frame_id, uid;
  int x1, y1, w, h;
};


struct TrackBox
{
  TrackBox(){}
  TrackBox(int f, int u, int x, int y, int w, int h):frame_id(f), uid(u), x1(x), y1(y), w(w), h(h){}
  void merge(TrackBox other){
    assert (this->frame_id == other.frame_id);
    x1 = (x1 + other.x1) / 2;
    y1 = (y1 + other.y1) / 2;
    w = (w + other.w) / 2;
    h = (h + other.h) / 2;
  }
  int frame_id, uid;
  int x1, y1, w, h;
};

struct BoxListInfo
{
  BoxListInfo(){}
  BoxListInfo(int t, int s, int e, int b):track_id(t), start_fid(s), end_fid(e), box_num(b){}
	int track_id, start_fid, end_fid, box_num;
};

struct BoxInfo {
  BoxInfo(){}
  BoxInfo(float c1, int f, int i, int x1, int y1, int x2, int y2, float c2):const_(c1),
                        frame_id(f), ind(i), x1(x1), y1(y1), x2(x2), y2(y2),const_1(c2){}
	float const_;
	int frame_id, ind, x1, y1, x2, y2;
	float const_1;
};

struct Edge{
  Edge(int a, int b):x1(a), x2(b){}
  int x1;
  int x2;
};

class MulticutTrack
{
public:
  MulticutTrack();
  void LoadTrackDat(const std::string& v_file, const std::string& e_file);
  bool MulticutProcess();
  bool DumpToFile(const std::string& frack_file);
  bool LoadTrack(const std::string& frack_file);
private:
  andres::graph::Graph<> m_graph;
  std::vector<float> m_weight_vector;
  std::vector<TrackBox> m_track_box;
  int num_vertical = 0;
  int num_edge = 0;
  std::map<int, std::vector<TrackBox>> m_track_info_map;
  std::vector<Edge> m_edge_vec;
  std::vector<float> m_edge_cos;
};

#include "MultiCutTrack.h"

MulticutTrack::MulticutTrack(){
  andres::graph::Graph<> x;
  m_graph = x;
}
/*
LoadTrackDat
detail: data should have the format: like
[the number of vertical]
[frm_id(int)x1(int)y1(int)w(int)h(int)] -- loop
[the number of edge]
[p1(int)p2(int)edge_cost(float)] -- loop
*/
void MulticutTrack::LoadTrackDat(const std::string& vertex_file, const std::string& edge_file){
  if(!file_exists(vertex_file) || !file_exists(edge_file)){
    return;
  }
  std::ifstream fin_v(vertex_file, std::ios::binary);
  std::ifstream fin_e(edge_file, std::ios::binary);
  fin_v.read((char*)&num_vertical, sizeof(num_vertical));
  fin_e.read((char*)&num_edge, sizeof(num_edge));
  std::cout<< "num_vertical " <<num_vertical << "\nnum_edge " << num_edge << std::endl;
  m_graph.insertVertices(num_vertical);
  for(int i=0; i < num_vertical; ++i){
    TrackBox temp;
    fin_v.read((char*)&temp, sizeof(temp));
    m_track_box.push_back(temp);
    //std::cout << "read frame id " << temp.frame_id << " uid "<< temp.uid << " read x1 y1 " << temp.x1 << " " << temp.y1 << std::endl;
  }
  for(int i = 0; i < num_edge; ++i){
    EdgeCost temp;
    fin_e.read((char*)&temp, sizeof(temp));
    m_graph.insertEdge(temp.start_vertical, temp.end_vertical);
    //insert edge
    //insert weight
    //m_edge_cost[Edge(temp.start_vertical, temp.end_vertical)] = temp.edge_cost;

    m_weight_vector.push_back(temp.edge_cost);
    m_edge_vec.push_back(Edge(temp.start_vertical, temp.end_vertical));
    m_edge_cos.push_back(temp.edge_cost);
    if(temp.start_vertical == 1162 && temp.end_vertical == 1164){
      std::cout<<temp.edge_cost<<" <-------------";
    }
  }
}
bool MulticutTrack::LoadTrack(const std::string& frack_file)
{
  std::ifstream fout(frack_file, std::ios::binary);
  while (!fout.eof())
	{
    BoxListInfo temp;
    fout.read((char*)&temp, sizeof(BoxListInfo));
    //std::cout << "track--id " << temp.track_id << std::endl;
    //std::cout << "frame--num " << temp.box_num << std::endl;
    for(int i=0; i < temp.box_num; ++i){
      BoxInfo box_info;
  		fout.read((char*)&box_info, sizeof(BoxInfo));
      //std::cout << i <<std::endl;
    }
    break;
  }
  return true;
}

bool MulticutTrack::DumpToFile(const std::string& frack_file)
{
  /*
  data = struct.pack("4i", track.track_id, track.start_fid, track.end_fid, track.box_num)
                f.write(data)
                for ind, b in enumerate(track._box_list):
                    x1, y1, x2, y2 = b.pos
                    data = struct.pack("f7i", 1.0, b.frame_id, ind, x1, y1, x2, y2, 1)
                    f.write(data)
  */
  assert (m_track_info_map.size() > 0);
  std::ofstream fin(frack_file, std::ios::out | std::ios::binary);
  for(std::map<int, std::vector<TrackBox>>::iterator it = m_track_info_map.begin(); it != m_track_info_map.end(); ++it) {
    int track_id = it->first;
    std::vector<TrackBox> track = it->second;
    if(track.size() <= k_track_len_thresh){
      continue;
    }
    sort( track.begin(), track.end(), [&]( const TrackBox& lhs, const TrackBox& rhs )
    {
       return lhs.frame_id <= rhs.frame_id;
    });
    std::vector<TrackBox> track_unique;
    track_unique.push_back(track[0]);
    int last_frame_id;
    //std::cout << "---------------------------------" <<std::endl;
    for(int i=1; i < track.size(); ++i)
    {
      last_frame_id = track[i].frame_id;
      if(track_unique.size() > 0){
        if(track_unique.back().frame_id == last_frame_id){
          track_unique.push_back(track[i]);
        }else{
          track_unique.push_back(track[i]);
        }

      }
    }
    int track_start_frm = track_unique.front().frame_id;
    int track_end_frm = track_unique.back().frame_id;
    int box_num = track_unique.size();
    BoxListInfo temp(track_id, track_start_frm ,track_end_frm ,box_num);
    fin.write((char*)&temp, sizeof(BoxListInfo));
    for(int i=0; i < track_unique.size(); ++i)
    {
      TrackBox box;
      box = track_unique[i];
      int frame_id = track_unique[i].frame_id;
      int x1, y1, x2, y2;
      x1 = track_unique[i].x1;
      y1 = track_unique[i].y1;
      x2 = track_unique[i].x1 + track_unique[i].w;
      y2 = track_unique[i].y1 + track_unique[i].h;
      BoxInfo box_info(1.0f, frame_id, i, x1, y1, x2, y2, 1.0f);
		  fin.write((char*)&box_info, sizeof(BoxInfo));
    }
  }
  fin.close();
  return true;
}
bool MulticutTrack::MulticutProcess(){
  printf("weight vector is %d\n", m_weight_vector.size());
  printf("num of edge is %d\n", m_graph.numberOfEdges());
  printf("num of vertex is %d\n", m_graph.numberOfVertices());
  std::map<int, std::vector<TrackBox>> track_info_map;
  std::vector<char> edge_labels(m_graph.numberOfEdges());

  std::vector<long unsigned int> vertex_labels(m_graph.numberOfVertices());
  vertex_labels = andres::graph::multicut::kernighanLin(m_graph, m_weight_vector, vertex_labels);
  //andres::graph::multicut::kernighanLin(m_graph, m_weight_vector, edge_labels, edge_labels);
  for(int i=0; i < m_edge_vec.size(); ++i){
    if (m_edge_vec[i].x1 == 93 and m_edge_vec[i].x2 == 95){
      std::cout << "fucking label is "<<int(edge_labels[i]) << std::endl;
    }
  }

  for(int i = 0; i < m_graph.numberOfVertices(); ++i){
    //printf("vertex is %d\n", vertex_labels[i]);
    unsigned int label = vertex_labels[i];
    if ( track_info_map.find(label) == track_info_map.end() ) {
      // not found
      std::vector<TrackBox> v;
      v.push_back(m_track_box[i]);
      track_info_map[label] = v;
      } else {
      // found
      track_info_map[label].push_back(m_track_box[i]);
    }
  }

  //print uid 0 track
  std::vector<TrackBox> track0 = track_info_map[0];
  for(std::vector<TrackBox>::iterator it = track0.begin(); it != track0.end(); ++ it){
    //std::cout << "frame id: "<< it->frame_id << " uid: " << it->uid << " x1: "<<it->x1 << " y1: " << it->y1 <<std::endl;
    for(std::vector<TrackBox>::iterator itt = track0.begin(); itt != track0.end(); ++ itt){
      if(it->x1 < 300 && itt->x1 >=300){
        for(int i=0; i < m_edge_vec.size(); ++i){
          if (m_edge_vec[i].x1 == it->uid and m_edge_vec[i].x2 == itt->uid){
            std::cout << "cost is "<<m_edge_cos[i] << std::endl;
          }
        }
      }
    }
  }

  std::vector<int> v;
  for(std::map<int, std::vector<TrackBox>>::iterator it = track_info_map.begin(); it != track_info_map.end(); ++it) {
    v.push_back(it->first);
    //std::cout << it->first << "\n";
  }
  m_track_info_map = track_info_map;

}

#include <queue>

using namespace std;


void StandardDBSCAN(int c, int n,
               int * X_core,
               int * neighbors,
               int * num_neighbors,
               int * core_or_not_indicators, 
               int * result) {
    /*
        Standard DBSCAN implementation, instead of the "DBSCAN_cy" which only clusters the core points.
        
        Parameters
        ----------
        c: Total number of core points in the original dataset
        n: Size of original dataset
        X_core: (c, ) array of the global indices of all the core points in the origianl dataset. 
                i.e., it only includes global indices of core points without their neighboring non-core points.
        neighbors: array of indices to each neighbor within eps_clustering
                   distance for all core points.
                   Note: unlike the parameter 'neighbors' in DBSCAN_cy, 
                        neighbors here include both core and non-core (i.e., cluster boundary) points, 
                        and each neighbor index in this array is global index (in the whole original dataset) 
                        instead of local index (in X_core) as in DBSCAN_cy.
        num_neighbors: (c, ) number of neighbors for each core point, to
                       be used for indexing into neighbors array
        core_or_not_indicators: (n, ) indicators for each point in the original dataset 
                                that indicate whether the point is core (>=0) or not (-1).
                                When the globally i-th point is core, 
                                core_or_not_indicators[i] is its local index in X_core, 
                                i.e., X_core[core_or_not_indicators[i]] == i.
                                When the globally i-th point is not core, 
                                core_or_not_indicators[i] == -1.
        result: (n, ) array of cluster results (i.e., cluster ID of each point) to be calculated
    */

    queue<int> q = queue<int>();
    int neighbor, start_ind, end_ind, point, cnt = 0;
    
    for (int i = 0; i < c; i++) {
        q = queue<int>();
        // this queue will only include core points, while non-core/noise points will never be pushed into it. 
        if (result[X_core[i]] == -1) {
            // if the current core point has not been visited, push its local index into queue. 
            q.push(i);

            while (!q.empty()) {
                // compute the cluster associated with the current core point
                point = q.front();
                q.pop();
                // note that 'point' can be either core or non-core

                start_ind = 0;
                if (point != 0) {
                    start_ind = num_neighbors[point - 1];
                }
                end_ind = num_neighbors[point];

                for (int j = start_ind; j < end_ind; j++) {
                    // Note: in DBSCAN_cy, the 'neighbors' records local indices (in X_core) of the neighbors of X_core;
                    //       but here, the 'neighbors' records global indices (in the original dataset) of the neighbors of X_core.
                    //       So in DBSCAN_cy, it uses 'X_core[neighbor]' to 
                    //       transform the local index 'neighbor' into global index of the current neighbor;
                    //       while here we directly use 'neighbor' since itself is already the global index.
                    neighbor = neighbors[j];
                    if (result[neighbor] == -1) {
                        // unvisited neighbor 
                        if (core_or_not_indicators[neighbor] >= 0) {
                            // the neighbor is core, so push its local index into the queue
                            q.push(core_or_not_indicators[neighbor]);
                        }
                        // non-core/noise points will never be pushed into the queue
                        
                        // assign the current cluster ID to the neighbor, 
                        // whether it is core or not.
                        result[neighbor] = cnt;
                    }
                }
            }

            cnt ++;
        }
    }
}

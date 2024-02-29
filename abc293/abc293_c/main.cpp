#include <iostream>
#include <set>
#include <algorithm>
using namespace std;

int main(void)
{
  int h, w;
  int a[11][11];
  
  cin >> h >> w;
  for(int y = 1; y <= h; y++) for(int x = 1; x <= w; x++) cin >> a[x][y];
  
  int p[20], l = h+w-2, ans = 0;
  for(int i = 1; i <= l; i++){
    if(i > w-1) p[i] = 1;
    else p[i] = 0;
  }
  for(auto x : p) cerr << x;
  cerr << endl;
  do{
    for(auto x : p) cerr << x;
    cerr << endl;
    int x = 1, y = 1;
    set<int> S; S.insert(a[1][1]);
    for(int i = 1; i <= l; i++){
      if(p[i]) y++;
      else x++;
      S.insert(a[x][y]);
    }
    if(S.size() == l+1) ans++;
  }while(next_permutation(p+1, p+l+1));
  
  cout << ans << endl;
  return 0;
}

#include <iostream>
#include <vector>
#include <queue>
using namespace std;
template<class T> using P = pair<T, T>;
int dx[] = { 1,0,-1,0,1,1,-1,-1 }; int dy[] = { 0,1,0,-1,1,-1,1,-1 };

int main() {
	int h, w;
	cin >> h >> w;
	vector<vector<char>> s(h, vector<char>(w));
	for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) cin >> s[i][j];
	int ans = 0;
	vector<vector<bool>> used(h, vector<bool>(w));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (s[i][j] == '.' or used[i][j]) continue;
			queue<P<int>> que;
			que.push({ i,j });
			while (!que.empty()) {
				P<int> p = que.front(); que.pop();
				int x = p.first, y = p.second;
				for (int d = 0; d < 8; d++) {
					int nx = x + dx[d];
					int ny = y + dy[d];
					if (0 <= nx && nx < h && 0 <= ny && ny < w && s[nx][ny] == '#' && !used[nx][ny]) {
						used[nx][ny] = true;
						que.push({ nx,ny });
					}
				}
			}
			ans++;
		}
	}
	cout << ans << endl;
}	

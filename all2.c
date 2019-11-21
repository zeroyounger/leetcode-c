//671
int top;
int findSecondMinimumValue(struct TreeNode* root){
    int *value = (int *)calloc(1024, sizeof(int));
    top = 0;
    find_node(value, root);
    unsigned int fir_min = 0xffffffff;
    unsigned int sec_min = 0xffffffff;
    for (int i = 0; i < top; ++i) {
        if (value[i] <= fir_min)     fir_min = value[i];
        else if (value[i] < sec_min) sec_min = value[i];
    }
    return (0xffffffff == sec_min ? -1 : sec_min);
}

int find_node(int value[], struct TreeNode* root){
    if (root == NULL) return 0;
    value[top++] = root->val;
    find_node(value, root->left);
    find_node(value, root->right);
    return 0;
}

//673
#define max(a,b) (((a) > (b)) ? (a) : (b))
int findNumberOfLIS(int* nums, int numsSize){
    int N=numsSize;
    if(N<=1) return N;
    int *lengths=(int*)calloc(N,sizeof(int));
    int counts[N];
    for(int i=0;i!=N;i++) counts[i]=1;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < j; ++i)
            if (nums[i] < nums[j])
                if (lengths[i] >= lengths[j]) {
                    lengths[j] = lengths[i] + 1;
                    counts[j] = counts[i];
                } else if (lengths[i] + 1 == lengths[j])
                    counts[j] += counts[i];
    int longest = 0, ans = 0;
    for (int i=0;i<N;i++)
        longest = max(longest, lengths[i]);
    for (int i = 0; i < N; ++i)
        if (lengths[i] == longest)
            ans += counts[i];
    return ans;
}

//674
int findLengthOfLCIS(int* nums, int numsSize){
    if(numsSize<2) return numsSize;
    int max=1, k=1;
    for(int i=0;i<numsSize-1;i++){
        if(nums[i]<nums[i+1]) k++;
        else k=1;
        if(k>max) max=k;
    }
    return max;
}

//680
bool isPalindrome(char* s,int l,int r){
    while(l < r){
        if(s[l] != s[r])
            return false;
        l++;
        r--;
    }
    return true;
}
bool validPalindrome(char* s) {
    int l = 0,r = strlen(s) - 1;
    while(l < r){
        if(s[l] != s[r])
            return isPalindrome(s,l+1,r) || isPalindrome(s,l,r-1);
        l++;
        r--;
    }
    return true;
}

//682
typedef struct{
    int *data;
    int top;
}stack;

stack* create_stack(int size){
    stack *p=(stack*)malloc(sizeof(stack));
    p->data = (int*)malloc(sizeof(int)*(size+1));
    p->top=0;
    return p;
}

int pop_stack(stack* p){
    if(p==NULL||p->top==0)
        return 0;
    int tem = p->data[p->top];
    p->top-=1;
    return tem;
}

void push_stack(stack* p,int x){
    p->data[++(p->top)]=x;
}

int top_stack(stack* p){
    return p->data[p->top];
}

int get_data(stack *p,int i){
    return p->data[p->top-(i-1)];
}

bool is_empty(stack* p){
    return p->top==0?true:false;
}

int get_int(char *str){
    int fg=0,ans=0;
    if(*str=='-')
        fg=1;
    for(int i=fg;str[i];i++)
        ans = ans*10+str[i]-'0';
    return fg?0-ans:ans;
}

int calPoints(char ** ops, int opsSize){
    stack *st = create_stack(opsSize);
    printf("%d",opsSize);
    int i=-1,score=0,tem=0;
    while(i++<opsSize-1){
        if(**(ops+i)=='D'){
            tem=top_stack(st)*2;
            score+=tem;
            push_stack(st,tem);
        }  
        else if(**(ops+i)=='+'){
            tem=top_stack(st)+get_data(st,2);
            score+=tem;
            push_stack(st,tem);
        }
            
        else if(**(ops+i)=='C'){
            score-=pop_stack(st);
        }
            
        else {
            tem=get_int(*(ops+i));
            score+=tem;
            push_stack(st,tem);}
    }
    return score;
}

//687
int global_max = 0;
int GetLength(struct TreeNode* b){
    if (b == NULL)
        return 0;
    int l = GetLength(b->left); 
    int r = GetLength(b->right);   
    if (b->left && b->val == b->left->val) 
        l++;
    else
        l = 0;
    if (b->right && b->val == b->right->val)
        r++;
    else 
        r = 0;
    if (l + r > global_max) 
        global_max = l + r; 
    return l > r ? l : r; 
}

int longestUnivaluePath(struct TreeNode* root){
    global_max = 0; //需要再次置0否则出错
    GetLength(root);
    return global_max;
}

//693
bool hasAlternatingBits(int n){
    int temp1,temp2 = -1;
    while(n>0){//将十进制n转为二进制
        temp1 = n%2;//指针1指向下二进制的下一位数
        if(temp1 == temp2)//两个指针不相等时候返回false
            return false;
        temp2 = temp1;//指针2指向指针1
        n=n/2;
    }
    return true;
}

//696
int countBinarySubstrings(char * s){
    int n = 0, pre = 0, curr = 1, len = strlen(s)-1;
    for (int i = 0; i < len; ++i) {
        if (s[i] == s[i+1]) ++curr;
        else{
            pre = curr; 
            curr = 1;
        }
        if (pre >= curr) ++n;
    }
    return n;
}

//697
int findShortestSubArray(int* nums, int numsSize){
int sum[50000]={0},first[50000],last[50000]={0};
    int i,min=50000,max=0;
    memset(first,-1,50000);
    for(i=0;i<numsSize;i++){
        sum[nums[i]]++;
        if(first[nums[i]]==-1)
            first[nums[i]]=i;
            last[nums[i]]=i;
         if(max<sum[nums[i]])
             max=sum[nums[i]];
    }
    for(i=0;i<numsSize;i++){
        if(sum[nums[i]]==max){
            if(min>last[nums[i]]-first[nums[i]]+1)
                min=last[nums[i]]-first[nums[i]]+1;
            if(min==max)
                return min;
        }
    }
    return min;
}

//700
struct TreeNode* searchBST(struct TreeNode* root, int val){
    if(!root) return NULL;
    if(root->val==val) return root;
    struct TreeNode *res=searchBST(root->left,val);
    if(res) return res;
    return searchBST(root->right,val);
}

//703
typedef struct {
    int max_size;
    int size_now;
    int *max;
    int min;
}KthLargest;
KthLargest *kthLargestCreate(int k, int *nums, int numsSize){
    int min = 0;
    int i = 0;
    KthLargest *largest = (KthLargest *)malloc(sizeof(KthLargest));
    largest->max_size = k;
    largest->max = (int *)malloc(sizeof(int) * (k + 1));
    largest->size_now = 0;
    if(numsSize!=0)
        largest->min = nums[0];
    for (i = 0; i < numsSize; i++) {
        kthLargestAdd(largest, nums[i]);
    }
    return largest;
}
void ModifyKthLargestFromHead(KthLargest *obj){
    int head = 0;
    int tmp;
    int change = 0;
    while (head < obj->size_now-1) {
        if(2*head+1 <= obj->size_now-1)
            change = 2*head+1;
        if(2*head+2 <= obj->size_now-1){
            change = obj->max[2*head+1]<obj->max[2*head+2]?(2*head+1):(2*head+2);
        }
        if((change <= obj->size_now-1) && (obj->max[head] > obj->max[change])){
            tmp = obj->max[change];
            obj->max[change] = obj->max[head];
            obj->max[head] = tmp;
            head = change;
            continue;
        }
        else
            break;
    }
    obj->min = obj->max[0];
    return;
}
void ModifyKthLargestFromTail(KthLargest *obj){
    int tail = obj->size_now-1;
    int tmp;
    while (tail >= 0) {
        if (obj->max[tail] < obj->max[(tail - 1) / 2]) {
            tmp = obj->max[(tail - 1) / 2];
            obj->max[(tail - 1) / 2] = obj->max[tail];
            obj->max[tail] = tmp;
            tail = (tail - 1) / 2;
        }
        else break;
    }
    obj->min = obj->max[0];
    return;
}
int kthLargestAdd(KthLargest *obj, int val){
    if (obj->size_now < obj->max_size) {
        obj->max[obj->size_now] = val;
        obj->size_now++;
        ModifyKthLargestFromTail(obj);
        return obj->min;
    }
    if (val > obj->min) {
        obj->max[0] = val;
        ModifyKthLargestFromHead(obj);
    }
    return obj->min;
}
void kthLargestFree(KthLargest *obj){
    if (obj) {
        if (obj->max) {
            free(obj->max);
        }
        free(obj);
    }
}

//704
int search(int* nums, int numsSize, int target){
    int l = 0, mid = 0;
    int r = numsSize -1;
    while (l <= r) {
        mid = ( l + r) / 2;
        if(nums[mid] == target) return mid;
        else if(nums[mid] < target) l = mid + 1;
        else r = mid -1;
    }
    return -1;
}

//706
#define HASH_SIZE 100000
typedef struct {
    int val[HASH_SIZE];
} MyHashMap;

/** Initialize your data structure here. */

MyHashMap* myHashMapCreate() {
    MyHashMap *hash = NULL;
    hash = (MyHashMap *)malloc(sizeof(MyHashMap));
    if (hash == NULL)
        return NULL;
    (void)memset((void *)hash, -1, sizeof(MyHashMap));
    return hash;
}

/** value will always be non-negative. */
void myHashMapPut(MyHashMap* obj, int key, int value) {
    int idx;
    if (obj == NULL)
        return;
    idx = key % HASH_SIZE;
    obj->val[idx] = value; 
    return;
}

/** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
int myHashMapGet(MyHashMap* obj, int key) {
    int idx;
    idx = key % HASH_SIZE;
    if(obj->val[idx] != -1)
        return obj->val[idx];
    return -1;
}

/** Removes the mapping of the specified value key if this map contains a mapping for the key */
void myHashMapRemove(MyHashMap* obj, int key) {
    int idx;
    if (obj == NULL)
        return;
    idx = key % HASH_SIZE;
    obj->val[idx] = -1;
    return;
}

void myHashMapFree(MyHashMap* obj) {
    if (obj == NULL) 
        return;
    free(obj);
    obj = NULL;
    return;
}

//709
char * toLowerCase(char * str){
    for(int i=0;i!=strlen(str);i++)
        str[i]|=32;
    return str;
}

//717
bool isOneBitCharacter(int* bits, int bitsSize){
    for (int i = 0; i < bitsSize; ++i) {
        if (i == bitsSize-1) return (bits[i] ? false : true);
        if (bits[i]) ++i;        
    }
    return false;
}

//720
int cmp(const void*a,const void*b) {
    char*aa = *(char**)a;
    char*bb = *(char**)b;
    if(strlen(aa)>strlen(bb)){
        return -1;
    } else if (strlen(aa) == strlen(bb)){
        if(strcmp(aa,bb)>0){
            return 1;
        } else {
            return -1;
        }
    } else {
        return 1;
    }
}
char * longestWord(char ** words, int wordsSize){
    qsort(words,wordsSize,sizeof(words[0]),cmp);
   for(int i=0;i<wordsSize-1;i++){
       int k=1;
       int ifok=1;
       char *temp=(char*)malloc(strlen(words[i])+1);
       while(k < strlen(words[i])){
           int have=0;
           memset(temp,0,strlen(words[i])+1);
           strncat(temp,words[i],strlen(words[i])-k);
           k++;
           for(int j=i+1;j<wordsSize;j++){
               if(strcmp(temp,words[j])==0){
                   have = 1;
                   break;
               }
           }
           if(have==1){
               continue;
           } else {
               ifok=0;
               break;
           }
       }
       if(ifok){
           return words[i];
       }
   }
    return words[wordsSize-1];
}

//724
int pivotIndex(int* nums, int numsSize) {
    int sum = 0,n = 0;
    for(int i = 0;i < numsSize;i++)
        sum = sum + nums[i];
    for(int i = 0;i < numsSize;i++){
        if(n * 2 == sum - nums[i])
            return i;
        n = n + nums[i];
    }
    return -1;
}

//728
int* selfDividingNumbers(int left, int right, int* returnSize) {
    int* a = calloc(right - left + 1,sizeof(int));
    int i = 0;
    *returnSize = 0;
    while(left <= right){
        int flag = 0,temp = left;
        while(temp > 0){
            int b = temp % 10;
            if(b == 0 || left % b != 0){
                flag = 1;
                break;
            }
            temp = temp / 10;
        }
        if(flag == 0){
            a[i] = left;
            i++;
            (*returnSize)++;
        }
        left++;
    }
    return a;
}

//733
void newDraw(int** image, int imageSize, int* imageColSize, int sr, int sc, int newColor, int flagColor){
    if (sr < 0 || sr >= imageSize || sc < 0 || sc >= imageColSize[sr]) return;
    if (image[sr][sc] != flagColor) return;
    image[sr][sc] = newColor;
    newDraw(image, imageSize, imageColSize, sr - 1, sc, newColor, flagColor);
    newDraw(image, imageSize, imageColSize, sr + 1, sc, newColor, flagColor);
    newDraw(image, imageSize, imageColSize, sr, sc - 1, newColor, flagColor);
    newDraw(image, imageSize, imageColSize, sr, sc + 1, newColor, flagColor);
}

int** floodFill(int** image, int imageSize, int* imageColSize, int sr, int sc, int newColor, int* returnSize, int** returnColumnSizes){
    int i;
    if (image == NULL || imageSize == 0 || imageColSize == NULL || returnSize == NULL || returnColumnSizes == NULL)
        return NULL;
    if (image[sr][sc] != newColor)
        newDraw(image, imageSize, imageColSize, sr, sc, newColor, image[sr][sc]);
    *returnSize = imageSize;
    *returnColumnSizes = imageColSize;
    return image;
}

//744
char nextGreatestLetter(char* letters, int lettersSize, char target){
    int left = 0;
    int right = lettersSize;
    int mid;
    while (left < right) {
        mid = left + (right - left) / 2;
        if (letters[mid] == target) {
            left = mid + 1;
        } else if (letters[mid] > target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return letters[left % lettersSize];
}

//746
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))
int minCostClimbingStairs(int* cost, int costSize){
    if (cost == NULL || costSize <= 0) return 0;
    if (costSize == 1) return cost[0];
    if (costSize == 2) return min(cost[0], cost[1]);
    int *dp = (int *)malloc(sizeof(int) * (costSize + 1));
    if (dp == NULL) return 0;
    memset(dp, 0, costSize + 1);
    dp[0] = cost[0]; // 从第一阶梯开始
    dp[1] = cost[1]; // 从第二阶梯开始
    for (int i = 2; i <= costSize; ++i) {
        if (i == costSize)
            dp[i] = min(dp[i - 2], dp[i - 1]);
        else {
            int oneChoice = dp[i - 2] + cost[i];
            int anotherChoice = dp[i - 1] + cost[i];
            dp[i] = min(oneChoice, anotherChoice) ;
        }
    }
    int result = dp[costSize];
    free(dp);
    dp = NULL;
    return result;
}

//747
int cmp(const void *a,const void *b){
    return (*(int *)a - *(int *)b);
}
int dominantIndex(int* nums, int numsSize){
    if(nums == 0 || numsSize == 0) return -1;
    if(numsSize == 1) return 0;
    int *tempsums = calloc(numsSize,sizeof(int));
    memcpy(tempsums,nums,numsSize*sizeof(int));
    qsort(tempsums,numsSize,sizeof(int),cmp);
    if(tempsums[numsSize -1 ] >= 2 * tempsums[numsSize-2]){
        for(int i = 0;i < numsSize;i++){
            if(nums[i] == tempsums[numsSize-1]){
                return i;
            }
        }
    }
    return -1;
}

//748
char * shortestCompletingWord(char * licensePlate, char ** words, int wordsSize){
    int *ht1=(int*)calloc(26,sizeof(int));
    for(int i=0;i!=strlen(licensePlate);i++)
        if(isalpha(licensePlate[i]))
            ht1[tolower(licensePlate[i])-'a']++;
    int min_idx=-1;
    for(int i=0;i!=wordsSize;i++){
        if(min_idx!=-1 && strlen(words[i])>=strlen(words[min_idx]))
            continue;
        int *ht2=(int*)calloc(26,sizeof(int));
        for(int j=0;j!=strlen(words[i]);j++){
            ht2[words[i][j]-'a']++;
        }
        int k=0;
        while(k<26 && ht1[k]<=ht2[k])
            k++;
        if(k>=26)
            min_idx=i;
    }
    return words[min_idx];
}

//754
int reachNumber(int target){
    int t=abs(target);
    int s=0;
    int dis=0;
    while(dis<t){
        s++;
        dis+=s;
    }
    int dt=dis-t;
    if(dt%2==0)
        return s;
    else{
        if(s%2==0)
            return s+1;
        else
            return s+2;
    }
}

//762
bool isPrime(int num) {
    if (num == 2) return true;
    if (num == 1 || (num & 0x1) == 0) return false;
    for (int i = 3; i <= sqrt(num); i += 2)
        if (num % i == 0) return false;
    return true;
}

int countCore(int num) {
    int count = 0;
    for(;num;count++)
        num &= (num - 1);
    return count;
}

int countPrimeSetBits(int L, int R){
    int count = 0;
    for (int i = L;i <= R;i++)
        if (isPrime(countCore(i)))
            count++;
    return count;
}

//766
bool isToeplitzMatrix(int** matrix, int matrixSize, int* matrixColSize){
    if(matrixSize==1) return true;
    for(int i=1;i<matrixSize;i++)
        for(int j=1;j<*matrixColSize;j++)
            if(matrix[i][j]!=matrix[i - 1][j - 1])
                return false;
    return true;    
}

//771
int numJewelsInStones(char * J, char * S){
    char hash[58] = {0};  // 'z' - 'A' + 1
    for (int i = 0; i < 50; i++) {
        if (J[i] == '\0') {
            break;
        }
        hash[J[i] - 'A'] = J[i];
    }
    int ret = 0;
    for (int i = 0; i < 50; i++) {
        if (S[i] == '\0') {
            break;
        }
        if (hash[S[i] - 'A'] == S[i]) {
            ret++;
        }
    }
    return ret;
}

//783
int min;
struct TreeNode* pre;
int minDiffInBST(struct TreeNode* root){
    min = 0x3f3f3f3f;
    pre = NULL;
    dfs(root);
    return min;
}

int dfs(struct TreeNode* root){
    if (root == NULL) return 0;
    dfs(root->left);
    if (pre != NULL) {
        if (min > (root->val - pre->val)) {
            min = root->val - pre->val;
        }
    }
    pre = root;
    dfs(root->right);
    return 0;
}

//784
#define MAX_STR_LEN       12
#define MAX_OUPUT_STR_NUM 20000
int g_Count = 0;
void Dfs(int index, char *S, int strLen, char **retStr){
    if (index >= strLen) {        
        retStr[g_Count] = (char *)malloc(sizeof(char) * (strLen + 1));                
         strcpy(retStr[g_Count], S);        
        g_Count++;
        return;
    }
    if (S[index] >= 'a' && S[index] <= 'z') {        
        S[index] -= 32;
        Dfs(index + 1, S, strLen, retStr);        
        S[index] += 32;
        Dfs(index + 1, S, strLen, retStr);        
    } else if (S[index] >= 'A' && S[index] <= 'Z') {       
        S[index] += 32;
        Dfs(index + 1, S, strLen, retStr);        
        S[index] -= 32;
        Dfs(index + 1, S, strLen, retStr);        
    } else {        
        Dfs(index + 1, S, strLen, retStr);        
    }
}
char ** letterCasePermutation(char * S, int* returnSize){
    if (S == NULL) {
        *returnSize = 0;
        return NULL;
    }
    int strLen = strlen(S);
    if (strLen > MAX_STR_LEN) {
        *returnSize = 1;
        return &S;
    }
    int alphaNum = 0;
    int memorySize = 1;
    for (int i = 0; i < strLen; i++) {
        if ((S[i] >= 'a' && S[i] <= 'z') 
           || (S[i] >= 'A' && S[i] <= 'Z')) {
            alphaNum++;
        }
    }
    while (alphaNum != 0) {
        memorySize *= 2;
        alphaNum--;
    }
    char **retStr = (char **)malloc(sizeof(char *) * memorySize);
    memset(retStr, 0, sizeof(char *) * memorySize);
    Dfs(0, S, strLen, retStr);
    *returnSize = g_Count;
    g_Count = 0;
    return retStr;
}

//788
int rotatedDigits(int N){
    int sum=0, sum1=0, num=0, count=0, flage=0;
    for (int i=2; i<=N; i++){
        num=i;
        while(num){
            sum=num%10;
            if (sum==3 || sum==4 || sum==7) break;
            if (sum==2 || sum==5 || sum==6 || sum==9)
                flage++;
            num/=10;
        }
        if (num==0 && flage>0) count++;
        flage=0;
    }
    return count;
}

//796
bool rotateString(char* A, char* B){
    int len = strlen(A);
    if((strlen(A) == 0) && (strlen(B) == 0))
        return true;
    int   i = 0;
    char* s = calloc(len + 1, sizeof(char));
    while(i < len){
        strncpy(s, A + i, len - i);  
        strncpy(s + len - i, A, i);
        i++;
        if(strcmp(s, B) == 0)
            return true;
    }
    return false;
}

//804
int uniqueMorseRepresentations(char ** words, int wordsSize){
    char m[][5]={".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};
    char s[wordsSize+1][50];
    int num=wordsSize,i,j;
    for (i=0;i<wordsSize;i++){    
        s[i][0]='\0';
        for (j=0;j<strlen(words[i]);j++)
            strcat(s[i],m[words[i][j]-'a']);
    }    
    for (i=0;i<wordsSize;i++) 
        for (j=i+1;j<wordsSize;j++)
            if (strcmp(s[i],s[j])==0){
                num--;
                break;
            }
    return num;
}

//806
int* numberOfLines(int* widths, int widthsSize, char * S, int* returnSize){
    int *res = malloc(sizeof(int) * 2);
    res[0] = 1;
    res[1] = 0;
    *returnSize = 2;
    int index, size = strlen(S);
    for (int i = 0; i < size; i++) {
        index = S[i] - 'a';
        if (res[1] + widths[index] > 100) {
            res[0]++;
            res[1] = widths[index];
        } else {
            res[1] += widths[index];
        }
    }
    return res;
}

//812
double area(double ax,double ay,double bx,double by,double cx,double cy) {
    double t=((bx-ax)*(cy-ay)-(cx-ax)*(by-ay))/2;
    return fabs(t);
}
double largestTriangleArea(int** points, int pointsSize, int pointsColSize) {
    double max,t;
    max=-1;
    for(int i=0; i<pointsSize; i++) {
        for(int j=i+1; j<pointsSize; j++) {
            for(int k=j+1; k<pointsSize; k++) {
                t=area(points[i][0],points[i][1],points[j][0],points[j][1],points[k][0],points[k][1]);
                if(t>max)
                    max=t;
            }
        }
    }
    return max;
}

//819
char * mostCommonWord(char * paragraph, char ** banned, int bannedSize){
    int len = strlen(paragraph);
    char word[1000][15];
    int num[1000];
    int temp = 0;
    int flag = 0;
    int maxTemp = 0;
    int maxNum = 0;
    for (int i = 0; i < len; i++) {
        if (paragraph[i] >= 'A' && paragraph[i] <= 'Z')
            paragraph[i] += 32;
        else if (paragraph[i] >= 'a' && paragraph[i] <= 'z')
            paragraph[i] = paragraph[i];
        else
            paragraph[i] = ' ';
    }
    for (int j = 0; j < bannedSize; j++) {
        int lenBan = strlen(banned[j]);
        char *position = strstr(paragraph, banned[j]);
        while (position != NULL) {
            if (*(position + lenBan) == ' ') {
                for (int k = 0; k < lenBan; k++)
                    position[k] = ' ';
                position = strstr(paragraph, banned[j]);
            }
            else
                position = strstr((position + 1), banned[j]);            
        }
    }
    int m = 0;
    int n = 0;
    for (int i = 0; i < len; i++) {
        if (paragraph[i] != ' ') {
            word[m][n] = paragraph[i];
            n++;
            flag = 1;
        } 
        else if (flag == 1) {
                word[m][n] = '\0';
                flag = 0;
                m++;
                n = 0;
        }
    }
    int p;
    for (p = 0; p < m; p++) {
        for (int k = p + 1; k < m; k++) {
            if (!strcmp(word[p], word[k]))
                temp++;
        }
        if (temp > maxTemp) {
            maxTemp = temp;
            maxNum = p;
        }
        temp = 0;
    }
    char *rest = (char*)malloc(strlen(word[maxNum]) + 1);
    strcpy(rest, word[maxNum]);
    return rest;
}

//821
#define min(a,b) (((a) < (b)) ? (a) : (b))
int* shortestToChar(char * S, char C, int* returnSize){
    int N=strlen(S);
    int *ans=(int*)calloc(N,sizeof(int));
    int prev=INT_MIN/2;
    for(int i=0;i!=N;++i){
        if(S[i]==C)
            prev=i;
        ans[i]=i-prev;
    }
    prev=INT_MAX/2;
    for(int i=N-1;i>=0;--i){
        if(S[i]==C)
            prev=i;
        ans[i]=min(ans[i],prev-i);
    }
    *returnSize=N;
    return ans;
}

//824
char * toGoatLatin(char * S){
    char *start = S;
    char *end;
    int cnt = 0;
    char * r = malloc(11250);
    memset(r, 0, 11250);
    int idx = 0;
    int len = strlen(S);
    for(int i = 0; i <= len; i++, S++){
        if( (*S == ' ') || (*S == '\0') ){   
            cnt++;
            end = S-1;
            if( 
                (*start == 'a') || (*start == 'e') || (*start == 'i') || (*start == 'o') || (*start == 'u') ||
                (*start == 'A') || (*start == 'E') || (*start == 'I') || (*start == 'O') || (*start == 'U')
            ){
                for(; start <= end; start++)
                    r[idx++] = *start;
                r[idx++] = 'm';
                r[idx++] = 'a';
            }
            else{
                char ch = *start;
                start++;
                for(; start <= end; start++)
                    r[idx++] = *start;
                r[idx++] = ch;
                r[idx++] = 'm';
                r[idx++] = 'a';
            }
            for (size_t j = 0; j < cnt; j++)
                r[idx++] = 'a';
            if(*S == ' ')
                r[idx++] = ' ';
            start = S+1;
        }
    }
    return r;
}

//830
int** largeGroupPositions(char * S, int* returnSize, int** returnColumnSizes){
    int** arr = (int**) malloc(strlen(S) / 3 * sizeof(int*));
    *returnSize = 0;
    int i, cnt = 1;
    for (i = 0; i < strlen(S); ++i) {
        if (S[i] == S[i + 1]) {
            ++cnt;
        } else {
            if (cnt >= 3) {
                arr[*returnSize] = (int*) malloc(2 * sizeof(int));
                arr[*returnSize][0] = i + 1 - cnt;
                arr[*returnSize][1] = i;
                ++(*returnSize);
            }
            cnt = 1;
        }
    }
    *returnColumnSizes = (int*) malloc(*returnSize * sizeof(int));
    for (i = 0; i < *returnSize; ++i)
        (*returnColumnSizes)[i] = 2;
    return arr;
}

//832
int** flipAndInvertImage(int** A, int ARowSize, int* AColSize, int* returnSize, int** returnColumnSizes){
   int **B = (int **)malloc(ARowSize * sizeof(int *));
    for (int i = 0; i < ARowSize; i++){
         B[i] = (int *)malloc(sizeof(int) * AColSize[i]);
        for (int j = 0; j < AColSize[i]; j++)
            B[i][j] = !A[i][AColSize[i] - j-1];
    }
    *returnSize = ARowSize;
    *returnColumnSizes = AColSize;
    return B;
}

//836
bool isRectangleOverlap(int* rec1, int rec1Size, int* rec2, int rec2Size){
    return !(rec1[2] <= rec2[0] ||   // left
                 rec1[3] <= rec2[1] ||   // bottom
                 rec1[0] >= rec2[2] ||   // right
                 rec1[1] >= rec2[3]);    // top
}

//840
bool Full(int **grid,int i,int j){
    int A[10]={0};
    for(int m=i;m-i<3;m++){
        for(int n=j;n-j<3;n++){
            if(grid[m][n]<1||grid[m][n]>9)
                return false;
            int k=grid[m][n];
            A[k]++;
        }
    }
    for(int k=1;k<10;k++){
        if(A[k]>1||A[k]==0)
            return false;
    }
    return true;
}

//判断行列以及对角线是否相等
bool Add(int **grid,int i,int j){
    int n1 = grid[i][j] + grid[i][j+1] + grid[i][j+2];
    int n2 = grid[i+1][j] + grid[i+1][j+1] + grid[i+1][j+2];
    int n3 = grid[i+2][j] + grid[i+2][j+1] + grid[i+2][j+2];
    int n4 = grid[i][j] + grid[i+1][j] + grid[i+2][j];
    int n5 = grid[i][j+1] + grid[i+1][j+1] + grid[i+2][j+1];
    int n6 = grid[i][j+2] + grid[i+1][j+2] + grid[i+2][j+2];
    int n7 = grid[i][j] + grid[i+1][j+1] + grid[i+2][j+2];
    int n8 = grid[i][j+2] + grid[i+1][j+1] + grid[i+2][j];
    if(n1 == n2 && n2 == n3 && n3 == n4 && n4 == n5 && n5 == n6 && n6 == n7 && n7 == n8)
         return true;
    else
         return false;
}

//判断是否为幻方
bool check(int **grid,int i,int j){
     if(grid[i+1][j+1]!=5)  //幻方中间数必须是5，此条件不满足则无需进行下一步判断
         return false;
     if(!Full(grid,i,j))
         return false;
     if(!Add(grid,i,j))
         return false;
     return true;
}

int numMagicSquaresInside(int** grid, int gridSize, int* gridColSize){
    if(gridSize<3||*gridColSize<3)
         return 0; //行数列数必须都大于3
    int count=0;
    for(int i=0;i<gridSize-2;i++){
        for(int j=0;j<*gridColSize-2;j++){
            if(check(grid,i,j))
                count++;
        }
    }
    return count;
}

//844
bool backspaceCompare(char * s_str, char * t_str){
    if (s_str == NULL && t_str == NULL) return true;
    if (s_str == NULL) return false;
    if (t_str == NULL) return false;
    
    int i = 0;
    int j = 0;
    while (s_str[j]) {
        if (s_str[j] == '#') {
            if (i > 0) --i;
        }else s_str[i++] = s_str[j];
        ++j;
    } 
    s_str[i] = 0;

    i=j=0;
    while (t_str[j]) {
        if (t_str[j] == '#')
            if (i > 0) --i;
        else 
            t_str[i++] = t_str[j];
        ++j;
    } 
    t_str[i] = 0;
    return !strcmp(s_str ,t_str);
}

//849
int max(int a, int b){
    return a > b ? a : b;
}

//001
typedef struct hash_node {
    int id;            /* we'll use this field as the key */
    int index;
    UT_hash_handle hh; /* makes this structure hashable */
} hash_node;

int* twoSum(int* nums, int numsSize, int target, int* returnSize){
    int *two_nums = (int *)malloc(sizeof(int)*2);
    hash_node *hash_table = NULL, *hash_item1 = NULL, *hash_item2 = NULL;
    for (int i = 0; i < numsSize; i++) {
        // 查找哈希表中是否存在满足和为target的另一个值,若存在直接返回
        int other_id = target - nums[i];
        HASH_FIND_INT(hash_table, &other_id, hash_item1);
        if (hash_item1) {
            two_nums[0] = hash_item1->index;
            two_nums[1] = i;
            *returnSize = 2;
            return two_nums;
        }
        // 将本次遍历的值放入哈希表,value为数组下标,key为对应数值
        hash_item2 = (hash_node *)malloc(sizeof(hash_node));
        hash_item2->id = nums[i];
        hash_item2->index = i;
        HASH_ADD_INT(hash_table, id, hash_item2);
    }
    return two_nums;
}

//002
struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2) {
    int c = 0;
    struct ListNode *head, *cur, *next;
    head = (struct ListNode *)malloc(sizeof(struct ListNode));
    head->next = NULL;
    cur = head;
    while(l1!=NULL || l2!=NULL || c) {
        next=(struct ListNode *)malloc(sizeof(struct ListNode));
        next->next=NULL;
        cur->next=next;
        cur=next;
        l1!=NULL?(c+=l1->val,l1=l1->next):(c+=0);
        l2!=NULL?(c+=l2->val,l2=l2->next):(c+=0);
        cur->val=c%10;
        c=c/10;
    }
    struct ListNode *del = head;
    head=head->next;
    free(del);
    return head;
}

//003
int lengthOfLongestSubstring(char * s){
    int prior = 0; //上次状态下最长子串的长度
    int dict[256] = {0}; //映射ASCII码
    int i;
    for (int left = 0, right = 1; *s != '\0'; right++){
        i = *s-0; //字符转换为整数
        if(dict[i] > left)    
            left = dict[i];
        dict[i] = right;
        prior = (prior>right-left)?prior:right-left; //right的值比对应的数组下标大1
        s++;
    }
    return prior;
}

//004
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
double findMedianSortedArrays(int* nums1, int nums1Size, int* nums2, int nums2Size){
    if (nums1Size>nums2Size)
        return findMedianSortedArrays(nums2, nums2Size, nums1, nums1Size);
    // Ci 为第i个数组的割,比如C1为2时表示第1个数组只有2个元素。LMaxi为第i个数组割后的左元素。RMini为第i个数组割后的右元素。
    int LMax1, LMax2, RMin1, RMin2, c1, c2, lo = 0, hi = 2 * nums1Size;//我们目前是虚拟加了'#'所以数组1是2*n长度
    while (lo <= hi){   //二分
        c1 = (lo + hi) / 2;  //c1是二分的结果
        c2 = nums1Size + nums2Size - c1;
        LMax1 = (c1 == 0) ? INT_MIN : nums1[(c1 - 1) / 2];
        RMin1 = (c1 == 2 * nums1Size) ? INT_MAX : nums1[c1 / 2];
        LMax2 = (c2 == 0) ? INT_MIN : nums2[(c2 - 1) / 2];
        RMin2 = (c2 == 2 * nums2Size) ? INT_MAX : nums2[c2 / 2];
        if (LMax1 > RMin2) hi = c1 - 1;
        else if (LMax2 > RMin1) lo = c1 + 1;
        else break;
    }
    return (max(LMax1, LMax2) + min(RMin1, RMin2)) / 2.0;
}

//005
#define max(a,b) (((a) > (b)) ? (a) : (b))
int expand(char * s, int l, int r){
    while(l>=0 && r<strlen(s) && s[l]==s[r]){
        l--;
        r++;
    }
    return r-l-1;
}
char * longestPalindrome(char * s){
    if (strlen(s)==0) return "";
    int umax = 1, umid = 0, size=strlen(s);
    for (int i=0; i<size-1; i++){
        int temp = max(expand(s,i,i+1), expand(s,i,i));
        if ((temp>umax) && temp>1){
            umid=i;
            umax=temp;
        }
    }
    int l=umid-(umax-1)/2;
    int r=umid+umax/2+1;
    char *ans = (char*)malloc(sizeof(char)*(r-l+1));
    memset(ans,0,r-l+1);
    strncpy(ans,s+l,r-l);
    return ans;
}

//007
int reverse(int x){
    int ans = 0;
    while(x!=0){
        int pop=x%10;
        if(ans>INT_MAX /10||(ans==INT_MAX/10 && pop>7)) return 0;
        if(ans<INT_MIN/10||(ans==INT_MIN/10 && pop<-8)) return 0;
        ans = ans*10+pop;
        x/=10;
    }
    return ans;
}

//009
bool isPalindrome(int x){
    int y = x ;  //用y记住原先值
    long res=0;  //直接定义为long型，用于存储翻转后的数据，因为int型取值范围内倒转可能会溢出
    while(x > 0){
         res = res*10 + x%10;   
         x = x/10;   
    }
    if(res == y) return true;
    return false;
}

//013
int mp(char c){
    switch(c){
        case 'I': return 1;
        case 'V': return 5;
        case 'X': return 10;
        case 'L': return 50;
        case 'C': return 100;
        case 'D': return 500;
        case 'M': return 1000;
        default: return 0;
    }
}

int romanToInt(char * s){
    int ans=0;
    for(int i=0;i<strlen(s)-1;i++){
        if(mp(s[i])<mp(s[i+1]))
            ans-=mp(s[i]);
        else
            ans+=mp(s[i]);
    }
    ans+=mp(s[strlen(s)-1]);
    return ans;
}

//014
int min(int x,int y,int z){
    int min=x;
    if (y<min) min=y;
    if (z<min) min=z;
    return min;
}
char * longestCommonPrefix(char ** strs, int strsSize){
    strsSize=0;
    char *str0=strs[0]; int len0=strlen(str0);
    char *str1=strs[1]; int len1=strlen(str1);
    char *str2=strs[2]; int len2=strlen(str2);
    int size=min(len0,len1,len2);
    for(int i=0;i!=size;i++){
        if(str0[i]==str1[i] && str0[i]==str2[i])
            strsSize++;
    }
    if(strsSize!=0){
        char *ans=(char*)malloc(sizeof(char)*strsSize);
        strncpy(ans,str0,strsSize);
        return ans;
    }
    return "";
}

//020
bool isValid(char * s){
    if (s == NULL || s[0] == '\0') return true;
    char *stack = (char*)malloc(strlen(s)+1); int top =0;
    for (int i = 0; s[i]; ++i) {
        if (s[i] == '(' || s[i] == '[' || s[i] == '{') stack[top++] = s[i];
        else {
            if ((--top) < 0)                      return false;//先减减，让top指向栈顶元素
            if (s[i] == ')' && stack[top] != '(') return false;
            if (s[i] == ']' && stack[top] != '[') return false;
            if (s[i] == '}' && stack[top] != '{') return false;
        }
    }
    return (!top);//防止“【”这种类似情况
}

//021
struct ListNode* mergeTwoLists(struct ListNode* l1, struct ListNode* l2){
    if(l1 == NULL) return l2;
    if(l2 == NULL) return l1;
    struct ListNode *head = (struct ListNode*)malloc(sizeof(struct ListNode));
    struct ListNode *p = head;
    while (l1 && l2){
        if (l1 -> val < l2 -> val){
            head -> next = l1;
            head = l1;
            l1 = l1 -> next; 
        }
        else{
            head -> next = l2;
            head = l2;
            l2 = l2 -> next;
        }
    }
    head -> next = l1 ? l1 : l2;
    return p -> next;
}

//026
int removeDuplicates(int* nums, int numsSize){
    if(numsSize==0) return 0;
    int i=0;
    for(int j=1;j<numsSize;j++)
        if(nums[j]!=nums[i]){
            i++;
            nums[i]=nums[j];
        }
    return i+1;
}

//027
int removeElement(int* nums, int numsSize, int val){
    int i = 0;
    for (int j = 0; j < numsSize; j++) {
        if (nums[j] != val) {
            nums[i] = nums[j];
            i++;
        }
    }
    return i;
}


//028 +
int strStr(char * haystack, char * needle){
    int len2 = strlen(needle);
    if (!len2) return 0;
    int len1 = strlen(haystack), j;
    for (int i = 0; i <= len1 - len2; i++)
        if (haystack[i] == needle[0] && haystack[i + len2 - 1] == needle[len2 - 1]){
            for (j = 1; j < len2; j++)
                if (haystack[i + j] != needle[j])
                    break;
            if (j == len2)
                return i;
        }
    return -1;
}

//035
int searchInsert(int* nums, int numsSize, int target){
    int l=0, r=numsSize-1, m;
    while(l<=r){
        m=l+((r-l)>>1);
        if(nums[m] == target) return m;
        if(target < nums[m]) r = m-1;
        if(target > nums[m]) l = m+1;
    }
    if(target <nums[m]) return m;
    return m+1;
}

//053
#define max(a,b) (((a)>(b))?(a):(b))
int maxSubArray(int* nums, int numsSize){
    int ans=nums[0];
    int sum=0;
    for(int i=0;i!=numsSize;i++){
        if(sum>0) sum+=nums[i];
        else sum=nums[i];
        ans=max(ans,sum);
    }
    return ans;
}

//058
int lengthOfLastWord(char * s){
    int len=strlen(s);
    if(len==0) return 0;
    int i=len-1;
    while(i>=0&&s[i]==' ') i--;
    len=i;
    while(i>=0&&s[i]!=' ') i--;
    return len-i;
}

//066
int* plusOne(int* digits, int digitsSize, int* returnSize){
    *returnSize=digitsSize;
    for(int i=digitsSize-1;i>=0;i--){
        digits[i]++;
        digits[i]%=10;
        if(digits[i]!=0) return digits;
    }
    *returnSize=digitsSize+1;
    digits=(int*)realloc(digits,*returnSize*(sizeof(int)));
    digits[0]=1;
    digits[digitsSize]=0;
    return digits;
}


//067
void Reverse(char *s,int n){ 
    for(int i=0,j=n-1;i<j;i++,j--){ 
        char c=s[i]; 
        s[i]=s[j]; 
        s[j]=c; 
    } 
}

char * addBinary(char * a, char * b){
    int c=0,i=strlen(a)-1,j=strlen(b)-1,k=0;
    char *s=(char*)calloc(i+j+3,sizeof(char));
    while(i >= 0 || j >= 0 || c == 1){
        c += i >= 0 ? a[i--] - '0' : 0;
        c += j >= 0 ? b[j--] - '0' : 0;
        s[k]=(c & 1) + '0';
        k++;
        c >>= 1;
    }
    Reverse(s,strlen(s));
    return s;
}

//069 牛顿迭代法
int mySqrt(int x){
    if(x<2)
        return x;
    double t = x;
    double x0 = x;
    x0 = x0/2 + t/(2*x0);
    while(x0*x0 - t > 0.00001)
        x0 = x0/2 + t/(2*x0);
    return (int)x0;
}

//070
int climbStairs(int n){
    if(n<=2)
        return n;
    int i1 = 1, i2 = 2;
    for(int i=3;i<=n;i++){
        int temp = i1+i2;
        i1 = i2;
        i2 = temp;
    }
    return i2;
}

//083
struct ListNode* deleteDuplicates(struct ListNode* head){
    struct ListNode *p, *next;
    p = next = head;
    while (p!=NULL) {
        while(next!=NULL && next->val==p->val) {
            next=next->next;
        }
        p->next=next;
        p=next;
    }
    return head;
}

//088
void merge(int* nums1, int nums1Size, int m, int* nums2, int nums2Size, int n){
    while(m&&n)
        nums1[m+n]=nums1[m-1]>nums2[n-1]?nums1[--m]:nums2[--n];
    while(n)nums1[n]=nums2[--n];
}

//091
int numDecodings(char * s) {
    if (s[0]=='0'||s[0]==0) return 0;
    int pre=1,curr=1;//dp[-1]=dp[0]=1
    for(int i=1;s[i]!=0;i++){
        int tmp=curr;
        if(s[i]=='0')
            if(s[i-1]=='1'||s[i-1]=='2') curr=pre;//dp[i]=dp[i-2]
            else return 0;
        else if(s[i-1]=='1'||s[i-1]=='2'&&s[i]>='1'&&s[i]<='6')
            curr=curr+pre;//dp[i]=dp[i-1]+dp[i-2]
        pre=tmp;
    }
    return curr;
}

//100
bool isSameTree(struct TreeNode* p, struct TreeNode* q){
    if(p == NULL && q == NULL) 
        return true;
    if(p == NULL || q == NULL) 
        return false;
    if(p->val != q->val) 
        return false;
    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

//101
bool isMirror(struct TreeNode* t1, struct TreeNode* t2) {
    if (t1 == NULL && t2 == NULL) return true;
    if (t1 == NULL || t2 == NULL) return false;
    return (t1->val == t2->val)
        && isMirror(t1->right, t2->left)
        && isMirror(t1->left, t2->right);
}

bool isSymmetric(struct TreeNode* root){
    return isMirror(root, root);
}

//104
int maxDepth(struct TreeNode* root){
    if(root == NULL) return 0;
    int left_length = maxDepth(root->left) + 1;
    int right_length = maxDepth(root->right) + 1;
    if( left_length >= right_length)
        return left_length;
    else
        return right_length;
}

//108
struct TreeNode* buildTree(int* nums, int l, int r){
    if (l>r) return NULL;
    int mid = l+(r-l)/2;
    struct TreeNode* root = (struct TreeNode*)malloc(sizeof(struct TreeNode));
    root->val=nums[mid];
    root->left=buildTree(nums,l,mid-1);
    root->right=buildTree(nums,mid+1,r);
    return root;
}
struct TreeNode* sortedArrayToBST(int* nums, int numsSize){
    return buildTree(nums, 0, numsSize - 1);
}

//110
#define max(a,b) (((a) > (b)) ? (a) : (b))
int depth(struct TreeNode* root){
    if(root==NULL) return 0;
    int left=depth(root->left);
    if(left==-1) return -1;
    int right=depth(root->right);
    if(right==-1) return -1;
    if(abs(left-right)<2)
        return max(left,right)+1;
    return -1;
}

bool isBalanced(struct TreeNode* root){
    return depth(root)!=-1;
}

//111
#define min(a,b) (((a) < (b)) ? (a) : (b))
int minDepth(struct TreeNode* root){
    if(root==NULL) return 0;
    int left=minDepth(root->left), right=minDepth(root->right);
    return (left && right) ? 1+min(left,right):1+left+right;
}

//112
bool hasPathSum(struct TreeNode* root, int sum){
    if(root==NULL) return false;
    if(root->left==NULL && root->right==NULL)
        return sum-root->val==0;
    return hasPathSum(root->left,sum-root->val)
        || hasPathSum(root->right,sum-root->val);
}

//118
int** generate(int numRows, int* returnSize, int** returnColumnSizes){
    *returnSize = numRows;
    int **nums = (int**)calloc(numRows,sizeof(int*));
    *returnColumnSizes = (int*)calloc(numRows,sizeof(int));//returnColumnSizes储存杨辉三角每一行元素的个数
    for (int i = 0; i < numRows; i++){
        (*returnColumnSizes)[i] = i+1;
        nums[i] = (int*)calloc(i+1,sizeof(int));
        for (int j=0; j<=i; j++)
            if (j==0 || j==i) 
                nums[i][j] = 1;
            else 
                nums[i][j] = nums[i - 1][j - 1] + nums[i - 1][j];
    }
    return nums;
}

//119
int* getRow(int rowIndex, int* returnSize) {
    *returnSize = rowIndex + 1;
    int* num = (int*) malloc ((rowIndex + 1) * sizeof(int));
    for(int i = 0;i <= rowIndex;i++)
        for(int j = i;j >= 0;j--){
            if(j == 0 || j == i)
                num[j] = 1;
            else
                num[j] = num[j] + num[j-1];
        }
    return num;
}

//121
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
int maxProfit(int* prices, int pricesSize){
    if(pricesSize <= 1)
        return 0;
    int in = prices[0], res = 0;
    for(int i = 1; i < pricesSize; i++) {
        res = max(res, prices[i] - in);
        in = min(in, prices[i]);
    }
    return res;
}

//122
int maxProfit(int *prices, int pricesSize){
    if (pricesSize == 0 && pricesSize == 1)
        return 0;
    int ret = 0;
    for (int i = 0; i < pricesSize - 1; i++) {
        if (prices[i] < prices[i + 1]) {
            ret = ret + prices[i + 1] - prices[i];
        }
    }
    return ret;
}

//125
bool isPalindrome(char * s){
    int len = strlen(s);
    if (len == 0)
        return true;
    int left = 0;
    int right = len - 1;
    while (left < right) {
        if (isalnum(s[left]) == false) {
            left++;
            continue;
        }
        if (isalnum(s[right]) == false) {
            right--;
            continue;
        }
        if(toupper(s[left]) == toupper(s[right])) {
            left++;
            right--;
            continue;
        }
        return false;
    }
    return true;
}

//128
typedef struct hash_node {
    int id;            /* we'll use this field as the key */
    int index;
    UT_hash_handle hh; /* makes this structure hashable */
} hash_node;
int max(int x,int y){
    return x>y?x:y;
}
int contain(hash_node *hash_table, int x){
    hash_node *hash_item1 = NULL;
    HASH_FIND_INT(hash_table, &x, hash_item1);
    if(hash_item1)
        return 1;
    return 0;
}
int longestConsecutive(int* nums, int numsSize){
    if(numsSize<1) return 0;
    hash_node *hash_table = NULL, *hash_item2 = NULL;
    for (int i=0;i!=numsSize;i++){
        hash_item2 = (hash_node *)malloc(sizeof(hash_node));
        hash_item2->id = nums[i];
        hash_item2->index = nums[i];
        HASH_ADD_INT(hash_table, id, hash_item2);
    }
    int longestStreak = 0;
    for (int i=0;i!=numsSize;i++){
        int num = nums[i];
        if(contain(hash_table,num-1)){
            int currentNum = num;
            int currentStreak = 1;
            while(contain(hash_table,currentNum+1)){
                currentNum++;
                currentStreak++;
            }
            longestStreak = max(longestStreak,currentStreak);
        }
    }
    return longestStreak+1;
}

//132
#define min(a,b) (((a) < (b)) ? (a) : (b))
int isPal(char * s, int size){
    int left=0,right=size-1;
    while(left<right)
        if(s[left++]!=s[right--])
            return 0;
    return 1;
}
int minCut(char * s){
    int n = strlen(s);
    int *f=(int*)malloc(sizeof(int)*(n+1));
    char *tmp=malloc(sizeof(char)*n);
    memset(tmp,0,n);
    f[0]=-1;
    for(int i=1;i<=n;i++){
        f[i]=i;
        for(int j=0;j<i;j++){
            strncpy(tmp,s+j,i-j);
            if(isPal(tmp,i-j))
                f[i]=min(f[j]+1,f[i]);
        }
    }
    free(tmp);
    return f[n];
}

//136
int singleNumber(int* nums, int numsSize){
    int res = nums[0];
    for(int i = 1; i < numsSize; i++ )
        res = res ^ nums[i];
    return res;
}


//141
bool hasCycle(struct ListNode *head) {
    if(head == NULL || head->next == NULL) return false;
    struct ListNode *pre, *p;
    pre = head; p = head->next;
    while(p&&p->next){
        if(pre==p) return true;
        pre = pre->next;
        p = p->next->next;
    }
    return false;
}

//160
struct ListNode *getIntersectionNode(struct ListNode *headA, struct ListNode *headB){//假设A比B长
    if(headA==NULL||headB==NULL)return NULL;
    struct ListNode* pA=headA;
    struct ListNode* pB=headB;
    while(pA!=pB){//遍历两个链表
        pA=pA==NULL?headB:pA->next;//构造链表D
        pB=pB==NULL?headA:pB->next;//构造链表C
    }
    return pA;
}

//167
int* twoSum(int* numbers, int numbersSize, int target, int* returnSize) {
    int i=0,j=numbersSize-1;
    *returnSize=2;
    while(i<j){
        if(numbers[i]+numbers[j]==target) break;
        else if(numbers[i]+numbers[j]<target) i++;
        else j--;
    }
    int* arr=(int*)malloc(sizeof(int)*2);
    arr[0]=i+1;
    arr[1]=j+1;
    return arr;
}

//168
char * convertToTitle(int n){
    int len = 0, tmp = n;
    while (tmp){
        len++;
        tmp = (tmp - 1) / 26;
    }
    char *res = (char*)malloc(len + 1);
    tmp = n;
    res[len] = 0;
    while (len--){
        res[len] = (tmp - 1) % 26 + 'A';
        tmp = (tmp - 1) / 26;
    }
    return res;
}

//169
int majorityElement(int* nums, int numsSize){
    int count = 1;
    int maj = nums[0];
    for (int i = 1; i < numsSize; i++) {
        if (maj == nums[i])
            count++;
        else {
            count--;
            if (count == 0) {
                maj = nums[i + 1];
            }
        }
    }
    return maj;
}

//171
int titleToNumber(char * s){
    int res = 0;
    for(int i = 0; i <strlen(s); ++i) res += (s[i]-'A'+1)*pow(26,strlen(s)-i-1);
    return res;
}

//172
int trailingZeroes(int n){
    if(n<5) return 0;
    return n/5+trailingZeroes(n/5);
}

//189
void rotate(int* nums, int numsSize, int k){
    int* res = (int*)malloc(sizeof(int)*numsSize);
    for(int i=0;i<numsSize;i++)
        res[(i+k)%numsSize] = nums[i];
    for(int i=0;i<numsSize;i++)
        nums[i] = res[i];
}

//190
uint32_t reverseBits(uint32_t n) { 
    n = ((n & 0xffff0000) >> 16) | ((n & 0x0000ffff) << 16); 
    n = ((n & 0xff00ff00) >>  8) | ((n & 0x00ff00ff) <<  8);  
    n = ((n & 0xf0f0f0f0) >>  4) | ((n & 0x0f0f0f0f) <<  4);  
    n = ((n & 0xcccccccc) >>  2) | ((n & 0x33333333) <<  2);  
    n = ((n & 0xaaaaaaaa) >>  1) | ((n & 0x55555555) <<  1);  
    return n;
}

//191
int hammingWeight(uint32_t n) {
    return (n > 0) ? 1 + hammingWeight(n & (n - 1)) : 0;
}

//198
#define max(a,b) (((a) > (b)) ? (a) : (b))
int rob(int* nums, int numsSize){
    if(numsSize==0) return 0;
    if(numsSize==1) return nums[0];
    int *dp=(int*)calloc(numsSize,sizeof(int));
    dp[0]=nums[0];
    dp[1]=nums[0]>nums[1]?nums[0]:nums[1];
    for(int i=2;i!=numsSize;i++)
        dp[i]=max(dp[i-2]+nums[i],dp[i-1]);
    return dp[numsSize-1];
}

//202
int step(int n){
    int sum = 0;
    while(n > 0){
        int bit=n%10;
        sum+=bit*bit;
        n/=10;
    }
    return sum;
}

bool isHappy(int n){
    int slow=n, fast=n;
    do{
        slow=step(slow);
        fast=step(fast);
        fast=step(fast);
    }while(slow!=fast);
    return slow==1;
}

//203
struct ListNode* removeElements(struct ListNode* head, int val){
    if(!head) return head;
    head->next = removeElements(head->next, val);
    return head->val == val ? head->next : head;    
}

//2043
int countPrimes(int n){
    if(n<3) return 0;
    bool *isPrime=(bool*)calloc(n+1, sizeof(bool));
    int count=0;
    for(int i=2;i!=n;++i){
        if(!isPrime[i]){
            count++;
            for(int j=i+i;j<n;j+=i)
                isPrime[j]=true;
        }
    }
    return count;
}

//205
bool isIsomorphic(char * s, char * t){        
    int a[128]={0};
    int b[128]={0};
    for(int i = 0; s[i]; ++i) {
        a[s[i]] += i+1;
        b[t[i]] += i+1;
        if(a[s[i]] != b[t[i]]) return false;
    }
    return true;
}

//206
struct ListNode* reverseList(struct ListNode* head){
    if (head == NULL) 
    if (head -> next == NULL) return head;
    struct ListNode *newHead = reverseList(head -> next);
    head -> next -> next = head;
    head -> next = NULL;
    return newHead;
}

//217
int cmp(const void *a, const void *b){
    return *((int*)a) > *((int*)b);
}
bool containsDuplicate(int* nums, int numsSize){
    int i;
    qsort(nums, numsSize, sizeof(int), cmp);
    for(i = 0; i < numsSize - 1; ++i)
        if(nums[i] == nums[i+1])
            return true;
    return false;
}

//219
bool containsNearbyDuplicate(int* nums, int numsSize, int k){
    if(numsSize<=1||numsSize>10000)
        return false;
    for(int i = 0;i<numsSize;i++){
        for(int j=i+1;j<=i+k&&j<numsSize;j++)
            if(nums[i]==nums[j])
                return true;
    }
    return false;
}

//226
struct TreeNode* invertTree(struct TreeNode* root){
    if(root==NULL) return NULL;
    struct TreeNode *rightTree=root->right;
    root->right=invertTree(root->left);
    root->left=invertTree(rightTree);
    return root;
}

//231
bool isPowerOfTwo(int n){
    if(n<=0) return false; 
    if((n&n-1)==0) return true; 
    return false;
}

//234
bool isPalindrome(struct ListNode* head){
    if(head == NULL || head->next == NULL){
        return true;
    }
    struct ListNode* mid1 = head;
    struct ListNode* tmp = head->next;
    while(tmp && tmp->next != NULL){
        mid1 = mid1->next;
        tmp = tmp->next->next;
    }
    struct ListNode* mid2 = mid1->next;
    struct ListNode* cur = mid2->next;
    while(cur != NULL){
        mid2->next = cur->next;
        cur->next = mid1->next;
        mid1->next = cur;
        cur = mid2->next;
    }
    mid2 = mid1->next;
    cur = head;
    while(mid2 != NULL){
        if(cur->val != mid2->val){
            return false;
        }
        cur = cur->next;
        mid2 = mid2->next;
    }
    return true;
}

//235
struct TreeNode* res=NULL;
struct TreeNode* lowestCommonAncestor(struct TreeNode* root, struct TreeNode* p, struct TreeNode* q){
    if((root->val-p->val)*(root->val-q->val)<=0)
        res=root;
    else if(root->val<p->val&&root->val<q->val)
        lowestCommonAncestor(root->right,p,q);
    else
        lowestCommonAncestor(root->left,p,q);
    return res;
}

//237
void deleteNode(struct ListNode* node) {
    *node = *(node->next);
    return;
}

//242
bool isAnagram(char *s, char *t){
    int i, x[26] = {0};
    for (i=0; s[i]!='\0'; i++)    x[s[i]-'a']++;
    for (i=0; t[i]!='\0'; i++)    x[t[i]-'a']--;
    for (i=0; i!=26; i++)
        if (x[i] != 0)
            return false;
    return true;
}

//257
char ** binaryTreePaths(struct TreeNode* root, int* returnSize){
    if (root == NULL) {
        * returnSize = 0;
        return NULL;
    }
    char **set = (char **) malloc(sizeof(char *) * 64);
    for (int i = 0; i < 64; ++i) {
        set[i] = (char *)malloc(sizeof(char) * 64);
    }
    *returnSize = 0;
    dfs_fun(root, set, returnSize, 0);
    ++(*returnSize);
    return set;
}
int dfs_fun(struct TreeNode* root, char **set, int *n, int i){
    int tmp;
    if (root->left != NULL && root->right != NULL) {//分支节点
        int tmp1 = *n;
        sprintf(&set[tmp1][i], "%d->", root->val);//防止值为负数
        i = strlen(set[tmp1]);//获得添加后字符串长度
        dfs_fun(root->left, set, n, i);
        ++(*n);//写到下一行字符串
        memcpy(set[*n], set[tmp1], i);
        dfs_fun(root->right, set, n, i);
    }else if (root->left != NULL) {
        sprintf(&set[*n][i], "%d->", root->val);
        i = strlen(set[*n]);        
        dfs_fun(root->left, set, n, i);
    }else if (root->right != NULL) {
        sprintf(&set[*n][i], "%d->", root->val);
        i = strlen(set[*n]);          
        dfs_fun(root->right, set, n, i);
    }else {//叶子节点
        sprintf(&set[*n][i], "%d", root->val);
    }
    return 0;
}

//258
int addDigits(int num){
    return (num-1)%9+1;
}

//263
bool isUgly(int num){
    if (num<1) return false;
    while (num%5==0) num/=5;
    while (num%3==0) num/=3;
    while (num%2==0) num>>=1;
    return num == 1;
}

//268
int missingNumber(int* nums, int numsSize){
    int res=numsSize;
    for (int i=0; i!= numsSize; ++i){
        res ^= nums[i];
        res ^= i;
    }
    return res;
}

//278
bool isBadVersion(int version);
int firstBadVersion(int n) {
    int l=1, h=n;
    while(l<h){
        int m=l+(h-l)/2;
        if (isBadVersion(m)) h=m;
        else l= m+1;
    }
    return h;
}

//283
void moveZeroes(int* nums, int numsSize) {
    int i = 0,j = 0;
    for(i = 0 ; i < numsSize; i++)
        if(nums[i] != 0)
            nums[j++] = nums[i];
    while(j < numsSize)
        nums[j++] = 0;
}

//290
bool wordPattern(char * pattern, char * str){
    char **hash = (char **)malloc(26 * sizeof(char*));
    for (int i = 0; i < 26; ++i){
        hash[i] = (char*)malloc(64 * sizeof(char));
        memset(hash[i], 0, 64 * sizeof(char));
    }
    int len = strlen(pattern);
    for (int i = 0; i < len; ++i){
        char *p = str;
        while (p && *p != 0 && *p != ' ') ++p;
        if (' ' == *p) *p++ = 0;
        if (strlen(str) == 0)
            return false;
        int pos = pattern[i] - 'a';
        if (strlen(hash[pos]) == 0){
            for (int j = 0; j < 26; ++j)
                if (j != pos && strlen(hash[j]) > 0
                    && strcmp(hash[j], str) == 0)
                        return false;
            strcpy(hash[pos], str);
        }
        else if (strcmp(hash[pos], str) != 0)
            return false;
        str = p;        
    }
    if (strlen(str) > 0)
        return false;
    return true;
}

//292
bool canWinNim(int n){
    return n%4;
}

//299
char g_result[10000] = {0};
#define min(a,b) (((a) < (b)) ? (a) : (b))
char * getHint(char * secret, char * guess){
    int aHashMap[10] = {0}, bHashMap[10] = {0};
    int i, aCount=0, bCount=0;
    for (i = 0; i < strlen(secret); i++)
        if (guess[i] == secret[i]) aCount++;
        else {
            aHashMap[secret[i] - '0']++;
            bHashMap[guess[i] - '0']++;
        }
    for (i = 0; i < 10; i++) 
        bCount += min(aHashMap[i], bHashMap[i]);
    sprintf(g_result, "%dA%dB", aCount, bCount);
    return g_result;
}

//312
int maxCoins(int* nums, int numsSize) {
    int *arr;
    int **dp;
    int i,j,k,step,tmp;
    arr = malloc(sizeof(int) * (numsSize + 2));
    dp = malloc(sizeof(int *) * (numsSize + 2));
    for(i = 0; i < numsSize + 2; ++i){
        dp[i] = malloc(sizeof(int) * (numsSize + 2));
        memset(dp[i], 0, sizeof(int) * (numsSize + 2));
    }
    arr[0] = arr[numsSize + 1] = 1;
    for(i = 1; i <= numsSize; ++i)
        arr[i] = nums[i-1];
    for(step = 1; step <= numsSize; ++step){
        for(i = 1; i <= numsSize - step + 1; ++i){
            j = i + step - 1;
            for(k = i; k <= j; ++k){
                tmp = dp[i][k-1] + arr[i-1] * arr[k] * arr[j+1] + dp[k+1][j];
                if(tmp > dp[i][j])
                    dp[i][j] = tmp;
            }
        }
    }
    tmp = dp[1][numsSize];
    free(arr);
    for(i = 0; i < numsSize + 2; ++i) free(dp[i]);
    free(dp);
    return tmp;
}

//326
bool isPowerOfThree(int n){
    return n > 0 && 1162261467%n == 0;
}

//342
bool isPowerOfFour(int num) {
    if (num < 0 || num & (num-1))//check(is or not) a power of 2.
        return false;
    return num & 0x55555555;//check 1 on odd bits
}

//344
void reverseString(char* s, int sSize){
    int l=0, r=sSize-1;
    char c;
    while(l<r){
        c=s[l];
        s[l++]=s[r];
        s[r--]=c;
    }
    return s;
}

//345
char * reverseVowels(char * s){
    int len = strlen(s);
    if (len <= 1)
        return s;
    char vowel[len + 1];
    int j = 0;
    for (int i = 0; i < len; i++){
        if (tolower(s[i]) == 'a' || tolower(s[i]) == 'e' || tolower(s[i]) == 'i' || 
            tolower(s[i]) == 'o' || tolower(s[i]) == 'u'){
            vowel[j++] = s[i];
            s[i] = '+';
        }
    }
    vowel[j] = '\0';
    for (int i = 0; i < len; i++){
        if (s[i] == '+')
            s[i] = vowel[--j];
    }
    return s;
}

//349
int* intersection(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize){
    int i, num=0;
    int *ret = (int*)calloc(5000,sizeof(int));
    bool *map=(bool*)calloc(5000,sizeof(bool));
    for(i=0; i<nums1Size; i++)
        map[nums1[i]] = 1;
    for(i=0; i<nums2Size; i++)
        if(map[nums2[i]] == 1){
            ret[num++] = nums2[i];
            map[nums2[i]] = 0;
        }
    *returnSize = num; 
    return ret;
}

//350
int cmp(const void *a, const void *b){
    return *(int*)a > *(int*)b;
}
int* intersect(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize){
    qsort(nums1, nums1Size, sizeof(int), cmp);
    qsort(nums2, nums2Size, sizeof(int), cmp);
    int  i=0, j=0, idx=0;
    int  min = (nums1Size > nums2Size) ? nums2Size : nums1Size;
    int* res = calloc(min,sizeof(int));
    while(i < nums1Size && j < nums2Size){
        if(nums1[i] < nums2[j]) i++;
        else if(nums1[i] > nums2[j]) j++;
        else{
            res[idx] = nums1[i];
            idx++;
            i++, j++;
        }
    }
    *returnSize = idx;
    return res;
}

//367
bool isPerfectSquare(int num){
    int sum=1;
    while(num>0){
        num-=sum;
        sum+=2;
    }
    return num==0;
}

//371
int getSum(int a, int b){
    while (b){
        unsigned int res = ((unsigned int ) (a & b))<<1 ; // 记录a+b的进位，直到进位为0是退出
        a = a^b;   //结果相加
        b = res;  //循环
    }
    return a;
}

//383
bool canConstruct(char * ransomNote, char * magazine){
    int hash[26] = {0};
    for(int i=0;magazine[i];++i)
        hash[magazine[i]-'a'] ++;
    for(int i=0;ransomNote[i];++i){
        hash[ransomNote[i]-'a'] --;
        if(hash[ransomNote[i] - 'a'] < 0)
            return false;
    }
    return true;
}

//386
int* lexicalOrder(int n, int* returnSize){
    int *res=(int*)calloc(n,sizeof(int));
    int cur=1;
    for(int i=0;i<n;i++){
        res[i]=cur;
        if(cur*10<=n){
            cur*=10;
        }else{
            if(cur>=n) cur/=10;
            cur+=1;
            while(cur%10==0) cur/=10;
        }
    }
    *returnSize=n;
    return res;
}

//387
int firstUniqChar(char * s){
    int len = strlen(s);
    int i = 0;
    if(len == 0) return -1;
    int table[26] = {0};
    for( i = 0; i < len;i++)
        table[s[i] - 'a']++;
    for(i = 0; i < len;i++){
        if(table[s[i] - 'a'] == 1)
            return i;
    }
    return -1;  
}

//389
char findTheDifference(char * s, char * t){
    char ret = s[0];
    for(int i=1; i< strlen(s); i++)
        ret ^= s[i];
    for(int i=0; i< strlen(t); i++)
        ret ^= t[i];
    return ret;
}

//392
bool isSubsequence(char * s, char * t){
    while(*s && *t){
        if(*s == *t) s++;
        t++;
    }
    if(*s=='\0') return true;
    return false;
}

//401
int bitsNum(int number) {
    int num = 0;
    while(number) {
        number = number & (number-1);
        num++;
    }
    return num;
}

char ** readBinaryWatch(int num, int* returnSize){
    *returnSize = 0;
    char **a = (char **)malloc(sizeof(char*)*3600);
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 60; j++) {
            if(bitsNum(i) + bitsNum(j) == num) {
                int k = *returnSize;
                if(i < 10) {
                    a[k] = malloc(sizeof(char)*5);
                    a[k][0] = i + '0';
                    a[k][1] = ':';
                    a[k][2] = j/10 + '0';
                    a[k][3] = j%10 + '0';
                    a[k][4] = '\0';                    
                } else {
                    a[k] = malloc(sizeof(char)*6);
                    a[k][0] = i/10 + '0';
                    a[k][1] = i%10 + '0';
                    a[k][2] = ':';
                    a[k][3] = j/10 + '0';
                    a[k][4] = j%10 + '0';
                    a[k][5] = '\0';
                }
                *returnSize = k+1;
            }
        }
    }
    return a;
}

//404
int sumOfLeftLeaves(struct TreeNode* root){
    if(root==NULL) return 0;
    int res=0;
    if(root->left!=NULL && root->left->left==NULL && root->left->right==NULL)
        res+=root->left->val;
    return sumOfLeftLeaves(root->left)+sumOfLeftLeaves(root->right)+res;
}

//405
char hex[9] = {0};
char * toHex(int num){
    sprintf(hex, "%x", num);
    return hex;
}

//409
int longestPalindrome(char * s){
    int sum = 0, len = strlen(s), table[58] = {0};
    while (*s) table[(*s++) - 'A']++;
    for (int i = 0; i < 58; ++i) sum += table[i] & 0xfffffffe;
    return len > sum ? ++sum : sum;
}

//412
char** fizzBuzz(int n, int* returnSize){
    char* fizz = malloc(5);
    char* buzz = malloc(5);
    char* fizzbuzz = malloc(9);
    sprintf(fizz, "%s", "Fizz");
    sprintf(buzz, "%s", "Buzz");
    sprintf(fizzbuzz, "%s", "FizzBuzz");
    char** res = (char**)malloc(sizeof(char*) * n);
    *returnSize = n;
    for (int i = 1; i <= n; i++) {
        if (i % 15 == 0)      res[i - 1] = fizzbuzz;
        else if (i % 5 == 0)  res[i - 1] = buzz;
        else if (i % 3 == 0)  res[i - 1] = fizz;
        else {
            res[i - 1] = malloc(10);
            sprintf(res[i - 1], "%d", i);
        }
    }
    return res;
}

//414
int thirdMax(int* nums, int numsSize){
    long long firMax=LLONG_MIN, secMax=LLONG_MIN, thiMax=LLONG_MIN;
    for(int i=0;i<numsSize;i++){
        if(nums[i]>firMax){
            thiMax=secMax;
            secMax=firMax;
            firMax=nums[i];
        }
        else if(nums[i]!=firMax&&nums[i]>secMax){
            thiMax=secMax;
            secMax=nums[i];
        }
        else if(nums[i]!=firMax&&nums[i]!=secMax&&nums[i]>thiMax)
            thiMax=nums[i];
    }
    return thiMax==LLONG_MIN? firMax:thiMax;
}

//415
char * addStrings(char * num1, char * num2){
    int len1 = strlen(num1);
    int len2 = strlen(num2);
    char *p1, *p2;
    if (len1 > len2) {
        p1 = num1 + len1 - 1;
        p2 = num2 + len2 - 1;
    } else {
        p2 = num1 + len1 - 1;
        p1 = num2 + len2 - 1;
        len1 = len1 ^ len2;
        len2 = len1 ^ len2;
        len1 = len1 ^ len2;
    }
    char *ret = (char*)malloc((len1 + 2) * sizeof(char));
    memset(ret, 0, (len1 + 2) * sizeof(char));
    int i, tmp, carry = 0;
    for (i = 0; i < len2; ++i) {
        tmp = *p1-- + *p2-- + carry - 96;
        ret[len1 - i] = tmp % 10 + 48;
        carry = tmp / 10;
    }
    for (i = len2; i < len1; ++i) {
        tmp = *p1-- + carry - 48;
        ret[len1 - i] = tmp % 10 + 48;
        carry = tmp / 10;
    }
    if (carry) {
        ret[0] = '1';
        return ret;
    }
    else
        return ret + 1;
}

//434
int countSegments(char * s){
    int len = strlen(s);
    int count = 0;
    for (int i = 0; i < len - 1; ++i)
        if (s[i] == ' ' && s[i + 1] != ' ') count++;
    if (count == len) return 0;
    return (s[0] == ' ') ? count : (count + 1);
}

//441
int arrangeCoins(int n){
    int i;
    long sum = 0;
    if(n == 1)
        return 1;
    for(i = 1; i < n; i++){
        sum += i;
        if(sum > n)
            break;
    }
    return i-1;
}

//443
int compress(char* chars, int charsSize){
    int i;
    int j;
    int len;
    int begin = 1;
    int count = 1;
    char num[10];
    for (i = 0; i < charsSize; i++) {
        if ((i != charsSize - 1) && (chars[i] == chars[i + 1])) {
            count++;
            continue;
        }
        if ((i == charsSize - 1) || (chars[i] != chars[i + 1])) {
            if (count > 1) {
                sprintf(num, "%d", count);
                len = strlen(num);
                for (j = 0; j < len; j++) {
                    chars[begin] = num[j];
                    begin++;
                }
                count = 1;
            }
            if (i != charsSize - 1) {
                chars[begin] = chars[i+1];
                begin++;           
            }
        }
    }
    return begin;
}

//447
int cmp(const void *a, const void *b){
    return *(int*)a - *(int*)b;
}
int numberOfBoomerangs(int** points, int pointsSize, int* pointsColSize){
    if (pointsSize < 2) return 0;
    int i, j, k, l, m, n, count = 0;
    int lineSize = pointsSize-1;
    int *lines = malloc(sizeof(int)*lineSize);
    for (i = 0; i < pointsSize; ++i) {
        k = 0;
        for (j = 0; j < pointsSize; ++j) {
            if (i == j) continue;
            lines[k++] = pow(points[i][0]-points[j][0], 2)+pow(points[i][1]-points[j][1], 2);
        }
        qsort(lines, lineSize, sizeof(int), cmp);
        m = lines[0];
        n = 1;
        for (l = 1; l < lineSize; ++l) {
            if (lines[l] != m) {
                if (n >= 2) count += n*(n-1);
                m = lines[l];
                n = 1;
            }else{
                ++n;
            }
        }
        if (n >= 2) count += n*(n-1);
    }
    
    free(lines);
    return count;
}

//448
int* findDisappearedNumbers(int* nums, int numsSize, int* returnSize){
    int i,j;
    for(i=0;i<numsSize;i++)
        nums[abs(nums[i])-1]=-abs(nums[abs(nums[i])-1]);    
    for(i=0,j=0;i<numsSize;i++)
        if(nums[i]>0)
            nums[j++]=i+1;    
    *returnSize=j;
    return nums;
}

//453
int minMoves(int* nums, int numsSize){
    long sum = 0,min = nums[0];
    for(int i = 0;i < numsSize;i++){
        sum = sum + nums[i];
        if(min > nums[i])
            min = nums[i];
    }
    return sum - min * numsSize;
}

//455
int cmp(const void* a, const void* b){
    return *(int*)a > *(int*)b;
}

int findContentChildren(int* g, int gSize, int* s, int sSize){
    qsort(g, gSize, sizeof(int), cmp);
    qsort(s, sSize, sizeof(int), cmp);
    int i = 0, j = 0, ans = 0;
    while(i < gSize && j < sSize){
        if(g[i] <= s[j]){
            ans++;
            i++;
            j++;
        }else
            j++;
    }
    return ans;
}


//459
bool repeatedSubstringPattern(char * s){
    int len = strlen(s);
    int *next = (int*)malloc((len + 1) * sizeof(int));
    memset(next, 0, (len + 1) * sizeof(int));
    next[0] = -1;
    int k = -1;
    int j = 0;
    while (j < len){
        if (k == -1 || s[j] == s[k]){
            ++j;
            ++k;
            next[j] = k;
        } else {
            k = next[k];
        }
    }
    return next[len] && len % (len - next[len]) == 0;
}

//461
int hammingDistance(int x, int y){
    int sum = x^y;
    int count = 0;
    while (sum){
        sum = sum & sum - 1;
        count++;
    }
    return count;
}

//475
int cmp(const void *a, const void *b){
    return *(int*)a - *(int*)b;
}

int findRadius(int* houses, int housesSize, int* heaters, int heatersSize){
    if (housesSize < 1 || heatersSize < 1) return 0;
    qsort(houses, housesSize, sizeof(int), cmp);
    qsort(heaters, heatersSize, sizeof(int), cmp);
    int i, d, pos = 0, radius = 0;
    for (i = 0; i < housesSize; ++i){
        while (pos < heatersSize - 1 && heaters[pos] < houses[i]) pos++;
        if (heaters[pos] <= houses[i]){
            d = houses[i] - heaters[pos];
            if (d > radius) radius = d;
        } else {
            d = heaters[pos] - houses[i];
            if (pos - 1 >= 0){
                int d2 = houses[i] - heaters[pos - 1];
                if (d2 < d) d = d2;
            }
            if (d > radius) radius = d;
        }
    }

    return radius;
}

//476
int findComplement(int num){
    unsigned int n = num;
    while ((n & (n - 1)) != 0) n &= (n - 1);
    n = (n << 1) - 1;
    return num ^ n;
}

//482
char * licenseKeyFormatting(char * S, int K){
    char *p, *p1=S, *p2=S;
    int n=0, len=0;
    while(*p1 != '\0'){
        if(*p1 != '-'){
            *p2++ = *p1;
            len++;
        }
        p1++;
    }
    *p2='\0';
    n=len%K;
    n=n?(K-n):0;
    p=malloc(len+len/K+1);
    p2=p;
    p1=S;
    while(*p1 != '\0'){
        if(n < K){
            *p2 = ((*p1 >= 'a') ? (*p1 - 32) : *p1);
            n++;
        }
        else{
            *p2++ ='-';
            *p2 = ((*p1 >= 'a') ? (*p1 - 32) : *p1);
            n=1;
        }
        p2++;
        p1++;
    }
    *p2='\0';
    return p;
}

//485
int findMaxConsecutiveOnes(int* nums, int numsSize){
    int max = 0;
    int count = 0;
    for(int i = 0; i < numsSize; i++){
        if(nums[i] == 1)
            count++;
        else{
            max = max > count ? max : count;
            count = 0;
        }
    }
    max = max > count ? max : count;
    return max;
}

//492
int* constructRectangle(int area, int* returnSize){
    int num = sqrt(area) + 1;
    while (area % num != 0) num--;
    *returnSize = 2;
    int* res = malloc(sizeof(int) * 2);
    int mid = area / num;
    if (mid > num) {
        int temp = num;
        num = mid;
        mid = temp;
    }
    res[0] = num;
    res[1] = mid;
    return res;
}

//496
int* nextGreaterElement(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize){
    *returnSize = nums1Size;
    if (nums1Size == 0) return NULL;
    int* res = malloc(sizeof(int) * nums1Size);
    for (int i = 0; i < nums1Size; i++) {
        int j = 0;
        while (nums1[i] != nums2[j]) j++;
        j++;
        while (j < nums2Size && nums2[j] <= nums1[i]) j++;
        if (j < nums2Size) res[i] = nums2[j];
        else res[i] = -1;
    }
    return res;
}

//500
char** findWords(char** words, int wordsSize, int* returnSize){
    char* line1 = "qwertyuiop";
    char* line2 = "asdfghjkl";
    char* line3 = "zxcvbnm";
    int  char_in_line_num = 0;
    char in_line_flag;
    char** r    = malloc(wordsSize * sizeof(char*));
    int    line = 0;
    for(int i = 0; i < wordsSize; i++){
        char* w      = words[i];
        int   pos    = 0;
        int*  record = malloc((strlen(w) + 1) * sizeof(int));
        in_line_flag = true;
        while(*w){  // 记录单词中每个字符所在的行
            char ch = *w;
            if(ch >= 'A' && ch <= 'Z')  // 大写变小写
                ch = *w + ('a' - 'A');
            if(strchr(line1, ch) != NULL)
                char_in_line_num = 1;
            else if(strchr(line2, ch) != NULL)
                char_in_line_num = 2;
            else if(strchr(line3, ch) != NULL)
                char_in_line_num = 3;
            else
                char_in_line_num = 10;
            record[pos++] = char_in_line_num;
            w++;
        }
        //检查每个字符所在行是否一致
        for(int j = 1; j < strlen(words[i]); j++){
            if(record[0] != record[j]){
                in_line_flag = false;
                break;
            }
        }
        if(in_line_flag){
            r[line] = malloc((strlen(words[i]) + 1) * sizeof(char));
            strcpy(r[line], words[i]);
            line++;
        }
        free(record);
    }
    *returnSize = line;
    return r;
}

//501
void findMaxSum(struct TreeNode* root, int *prev, int *curr_size, int* max_size, int* res, int* returnSize){
    if (NULL == root) return;
    findMaxSum(root->left, prev, curr_size, max_size, res, returnSize);
    if (*prev == root->val) {     
        *curr_size += 1;      
    } else {
        *curr_size = 1;
        *prev = root->val;
    }
    if (*curr_size == *max_size){
        res[(*returnSize)++] = *prev;
    }
    if (*curr_size > *max_size){
        *max_size = *curr_size;
        *returnSize = 0;
        res[(*returnSize)++] = *prev;
    }
    findMaxSum(root->right, prev, curr_size, max_size, res, returnSize);
}
int* findMode(struct TreeNode* root, int* returnSize){
    *returnSize = 0;
    if (NULL == root) return NULL;
    int *res = (int*)malloc(10240 * sizeof(int));
    int max_size = -1;
    int curr_size = 0;
    int prev = root->val;
    findMaxSum(root, &prev, &curr_size, &max_size, res, returnSize);
    return res;
}

//504
char * convertToBase7(int num){
    char *res = (char*)malloc(13 * sizeof(char));
    memset(res, 0, 13 * sizeof(char));
    int i, index = 0;
    int tmp = num >= 0 ? num : - num;
    while (tmp) {
        res[index++] = tmp % 7 + '0';
        tmp /= 7;
    }
    if (num < 0) res[index++] = '-';
    if (0 == num) res[index++] = '0';
    for (i = 0; i < index / 2; ++i) {
        tmp = res[i];
        res[i] = res[index - 1 - i];
        res[index - 1- i] = tmp;
    }
    return res;
}

//506
typedef struct _Data{
    int index;
    int val;
} Data;

int cmp(const void *a, const void *b){
    return ((Data*)b)->val - ((Data*)a)->val;
}

char ** findRelativeRanks(int* nums, int numsSize, int* returnSize){
    *returnSize = numsSize;
    if (numsSize == 0) return NULL;
    Data *d = (Data*)malloc(numsSize * sizeof(Data));
    int i;
    for (i = 0; i < numsSize; ++i) {
        d[i].index = i;
        d[i].val = nums[i];
    }
    qsort(d, numsSize, sizeof(Data), cmp);
    char **res = (char**)malloc(numsSize * sizeof(char*));
    memset(res, 0, numsSize * sizeof(char*));
    for (i = 0; i < numsSize; ++i){
        res[d[i].index] = (char*)malloc(13 * sizeof(char));
        memset(res[d[i].index], 0, 13 * sizeof(char));
        switch (i) {
        case 0:
            sprintf(res[d[i].index], "Gold Medal");
            break;
        case 1:
            sprintf(res[d[i].index], "Silver Medal");
            break;
        case 2:
            sprintf(res[d[i].index], "Bronze Medal");
            break;
        default:
            sprintf(res[d[i].index], "%d", i + 1);
            break;
        }
    }
    return res;
}

//507
bool checkPerfectNumber(int num){
    if (0 == num || (num & 1)) return false;
    int i, sum = 1, sq = sqrt(num);
    for (i = 2; i < sq; ++i) if (num % i == 0) sum += i + num / i;
    if (num % sq == 0) sum += (num / sq == sq) ? sq : sq + num / sq;
    return sum == num;
}

//509
int fib(int N){
    if (N == 0)
        return 0;
    if (N == 1)
        return 1;
    return fib(N - 1) + fib(N - 2);
}

//520
bool detectLetter(char letter){
    if (letter >= 'A' && letter <= 'Z')
        return true;
    else if (letter >= 'a' && letter <= 'z')
        return false;
    return false;
}

bool detectCapitalUse(char* word){
    if (!word[1])
        return true;
    if (detectLetter(word[0]) == false && detectLetter(word[1]) == true)
        return false;
    bool flag = detectLetter(word[1]);
    int i = 2;
    while (word[i]&&word[i]!='\0'){
        if (detectLetter(word[i]) != flag)
            return false;
        i++;
    }
    return true;
}

//521
int findLUSlength(char * a, char * b){
    if (0 == strcmp(a, b))
        return -1;
    if (strlen(a) >= strlen(b))
        return strlen(a);
    return strlen(b);
}

//530
void GetMid(struct TreeNode* root, int *ans, int *size){
    if (root->left != NULL)
        GetMid(root->left, ans, size);
    ans[(*size)++] = root->val;
    if (root->right != NULL)
        GetMid(root->right, ans, size);
}

int getMinimumDifference(struct TreeNode* root){
    int midSeq[10000] = {0};
    int size = 0;
    GetMid(root, midSeq, &size);
    int ans = INT_MAX;
    for (int i = 1; i < size; i++) {
        int tmp = midSeq[i] - midSeq[i - 1];
        if (tmp < ans)
            ans = tmp;
    }
    return ans;
}

//532
int cmp(const void *a, const void *b){
    return *(int*)b - *(int*)a;
}

int findPairs(int* nums, int numsSize, int k){
    if (numsSize < 2 || k < 0) return 0;
    qsort(nums, numsSize, sizeof(int), cmp);
    int i, j, curr, count = 0, prev = 0x80000000;
    for (i = 0; i < numsSize - 1; ++i){
        if (nums[i] != prev) {
            prev = nums[i];
            curr = nums[i] - k;
            for (j = i + 1; j < numsSize && nums[j] >= curr; ++j) {
                if (nums[j] == curr) {
                    count++;
                    break;
                }
            }
        }
    }
    return count;
}

//538
void accBST(struct TreeNode *root, int *sum){
    if (NULL == root) return;
    accBST(root->right, sum);
    *sum += root->val;
    root->val = *sum;
    accBST(root->left, sum);
}

struct TreeNode* convertBST(struct TreeNode* root){
    int sum = 0;
    accBST(root, &sum);
    return root;
}

//541
char * reverseStr(char * s, int k){
    int len = strlen(s);
    for (int i = 0; i < len; i+=k*2) { 
        if (i+k <= len) overturn(s, i, i+k);
        else            overturn(s, i, len);
    }
    return s;
}

void overturn(char * str, int fast, int last){
    int i = fast;
    int j = last - 1;
    while (i < j) {
        str[i]   ^= str[j];
        str[j]   ^= str[i];
        str[i++] ^= str[j--];
    }
}

//543
int calMaxRoot(struct TreeNode* root, int *maxRoot){
    if (NULL == root) return 0;
    int lLen = calMaxRoot(root->left, maxRoot);
    int rLen = calMaxRoot(root->right, maxRoot);
    if (lLen + rLen > *maxRoot) *maxRoot = lLen + rLen;
    return (lLen > rLen ? lLen : rLen) + 1;
}

int diameterOfBinaryTree(struct TreeNode* root){
    int maxRoot = 0;
    calMaxRoot(root, &maxRoot);
    return maxRoot;
}

//557
char * reverseWords(char * str){
    if (str == NULL || str[0] == '\0') return str;
    int fast = 0, last = 0;
    while (str[fast] && str[fast] == ' ') ++fast; //找字符第一个非空格字符
    while (str[fast]) {
        for (last = fast; str[last] && str[last] != ' '; ++last); //找单词的最后一个字符
        for (int i = fast, j = last-1; i < j; ++i, --j) str[i] ^= str[j] ^= str[i] ^= str[j];//交换
        for (fast = last; str[fast] && str[fast] == ' '; ++fast); //找单词的第一个字符
    }
    return str;
}

//563
int NODE_SUMS[10240] = {0};
int NODE_NUMBER = 0;
int findTilt(struct TreeNode* root){
    NODE_NUMBER = 0;
    for (int i = 0; i < 1024; ++i) NODE_SUMS[i] = 0;
    node_dfs(root);
    int sum = 0;
    for (int i = 0; i < NODE_NUMBER; i++)
        sum += NODE_SUMS[i];
    return sum;
}

int node_dfs(struct TreeNode* root){
    if (root == NULL) return 0;
    int left_val = node_dfs(root->left);
    int right_val = node_dfs(root->right);
    int tmp = left_val - right_val;
    tmp = (tmp > 0 ? tmp : -tmp);
    NODE_SUMS[NODE_NUMBER++] = tmp;//装入缓冲栈中
    return root->val + left_val + right_val;
}

//572
bool isSame(struct TreeNode* s, struct TreeNode* t){
    if (s == NULL && t == NULL) return true;
    if (s == NULL || t == NULL) return false;
    if (s->val == t->val) return isSame(s->left, t->left) && isSame(s->right, t->right);
    return false;
}
bool isSubtree(struct TreeNode* s, struct TreeNode* t){
    if (s == NULL && t == NULL) return true;
    if (s != NULL && t == NULL) return true;
    if (s == NULL) return false;
    return (isSame(s,t) || isSubtree(s->left, t) || isSubtree(s->right, t));
}

//575
int distributeCandies(int* candies, int candiesSize){
    char map[25001] = {0};
    int temp = 0,count = 0;
    int len = 0;
    for(int i = 0;i<candiesSize;i++){
        temp = (candies[i] + 100000);//将数缩放到我们数组下标的范围
        map[temp/8] |= 1 << (temp%8); //数应该在第temp/8+1个字节(即map[temp/8])的第(temp%8)位，使用位运算将标志写入对应的位
    }
    for(int i = 0;i<25001;i++){//输入完毕，开始统计有多少种糖果
        for(int j = 0;j<8;j++){
            count += (map[i]>>j) & 1;//读取每一位
        }
    }
    return count < candiesSize/2? count:candiesSize/2;
}

//581
int findUnsortedSubarray(int* nums, int numsSize){
    if (numsSize <= 1)
        return 0;
    int maxNum = nums[0];
    int minNum = nums[numsSize - 1];
    int left;
    int right;
    for(int i=0; i<numsSize; i++){
        if(nums[i]>=maxNum){
            maxNum = nums[i];
            continue;
        }
        left = i;
    }
    for(int i=numsSize-1; i>=0; i--){
        if(nums[i]<=minNum){
            minNum = nums[i];
            continue;
        }
        right = i;
    }
    return left>right ? left-right+1 : 0;
}

//583
int max(int x, int y){
    return x>y?x:y;
}
int minDistance(char * s1, char * s2) {
    int m = strlen(s1), n = strlen(s2);
    int (*dp)[n+1]=malloc(sizeof(int)*(m+1)*(n+1));
    for(int i=0;i!=m+1;i++)
        for(int j=0;j!=n+1;j++)
            dp[i][j]=0;
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(s1[i-1]==s2[j-1]){
                dp[i][j]=dp[i-1][j-1]+1;
            }else{
                dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
            }
        }
    }
    return m + n - 2 * dp[m][n];
}

//594
int cmp(const void *a, const void *b){
    return *(int *)a - *(int *)b;
}

int max(int a, int b){//找两个数中较大的数
    if(a > b) return a;
    else return b;
}

int findLHS(int* nums, int numsSize){
    if (numsSize < 2) return 0;
    qsort(nums, numsSize, sizeof(int), cmp);
    int start = 0;
    int end;
    int ret = 0;
    for (end = 0; end < numsSize; end++) {
        while (nums[end] - nums[start] > 1) start++;
        if (nums[end] - nums[start] == 1)
            ret = max(ret, end - start + 1);
    }
    return ret;
}

//598
int maxCount(int m, int n, int** ops, int opsSize, int* opsColSize){
    int i = m;
    int j = n;
    if(ops == NULL || opsSize == 0){
        return i*j;
    }
    int idx = 0;
    for(idx = 0; idx<opsSize; idx++){
        i = i < ops[idx][0] ? i : ops[idx][0];
        j = j < ops[idx][1] ? j : ops[idx][1];
    }
    return i*j;
}

//599
struct hash_node{
    char *key;
    int val;
    struct hash_node* next;
};
struct hash_table{
    struct hash_node** head;
    int hash_width;
};
int hash_init(struct hash_table* table, int hash_width){
    if (hash_width <= 0) return -1;
    table->head = (struct hash_node**)malloc(hash_width * sizeof(struct hash_node*));
    if (!table->head) return -1;
    memset(table->head, 0, hash_width * sizeof(struct hash_node*));
    table->hash_width = hash_width;
    return 0;
}
void hash_free(struct hash_table* table){
    if (table->head){
        for (int i = 0; i < table->hash_width; ++i){
            struct hash_node* p = table->head[i];
            while (p){
                struct hash_node* tmp = p;
                p = p->next;
                free(p);
            }
        }
        free(table->head);
        table->head = NULL;
    }
    table->hash_width = 0;
}
int hash_addr(int hash_width, char *key){
    int k = 0;
    while (*key) k += *key++;
    return k % hash_width;
}
int hash_insert(struct hash_table* table, char* key, int val){
    int k = hash_addr(table->hash_width, key);
    struct hash_node* p = (struct hash_node*)malloc(sizeof(struct hash_node));
    if (!p) return -1;
    p->key = key;
    p->val = val;
    p->next = table->head[k];
    table->head[k] = p;
    return 0;
}
struct hash_node* hash_find(struct hash_table* table, char* key){
    int k = hash_addr(table->hash_width, key);
    struct hash_node* p = table->head[k];
    while (p){
        if (0 == strcmp(p->key, key)) return p;
        p = p->next;
    }
    return NULL;
}
char** findRestaurant(char** list1, int list1Size, char** list2, int list2Size, int* returnSize){
    char** res = NULL;
    *returnSize = 0;
    int i, minVal = list1Size + list2Size;
    struct hash_table table;
    hash_init(&table, 100);
    for (i = 0; i < list1Size; ++i){
        hash_insert(&table, list1[i], i);
    }
    for (i = 0; i < list2Size; ++i){
        struct hash_node* p = hash_find(&table, list2[i]);
        if (p) {
            if (i + p->val < minVal){
                minVal = i + p->val;               
                res = (char**)realloc(res, sizeof(char*));
                *returnSize = 0;
                res[(*returnSize)++] = list2[i];
            } else if (i + p->val == minVal){
                res = (char**)realloc(res, (1 + *returnSize) * sizeof(char*));
                res[(*returnSize)++] = list2[i];
            }
        }
    }
    return res;
}

//600
int findIntegers(int num){
    int f[32];
    f[0]=1;
    f[1]=2;
    for(int i=2;i<32;i++)
        f[i]=f[i-1]+f[i-2];
    int i=30,sum=0,prev_bit=0;
    while(i>=0){
        if((num & (1<<i))!=0){
            sum+=f[i];
            if(prev_bit==1){
                sum--;
                break;
            }
            prev_bit=1;
        }
        else
            prev_bit=0;
        i--;
    }
    return sum+1;
}

//605
bool canPlaceFlowers(int* flowerbed, int flowerbedSize, int n){
    int i, len0=0, num=0;
    if(flowerbedSize==1&&flowerbed[0]==0) return true;
    for(i=0;i<flowerbedSize;i++){
        if(flowerbed[i]==0){
            len0++;
            if((i==1||i==flowerbedSize-1) && len0>=2){
                num++;
                len0=1;
            }
            else if(len0==3){
                num++;
                len0=1;   
            }  
        }
        else len0=0;
    }
    if(num>=n) return true;
    else return false;
}

//617
struct TreeNode* mergeTrees(struct TreeNode* t1, struct TreeNode* t2){
    if (!t1){return t2;}
    if (!t2){return t1;}
    t1->val +=t2->val;
    t1->left=mergeTrees(t1->left,t2->left);
    t1->right=mergeTrees(t1->right,t2->right);
    return t1;
}

//628
#define max(a,b) (((a) > (b)) ? (a) : (b))
int cmp(const void *a, const void *b){ return *(int*)a > *(int*)b;}
int maximumProduct(int* nums, int numsSize){
    qsort(nums, numsSize, sizeof(nums[0]), cmp);
    return max(nums[0]*nums[1]*nums[numsSize-1],nums[numsSize-3]*nums[numsSize-2]*nums[numsSize-1]);
}

//633
bool judgeSquareSum(int c){
    double L = sqrt(c);
    for(int i=0;i<=L;i++){
        double a=sqrt(c-i*i);
        if((int)a==a)
            return true;
    }
    return false;
}

//637
#define MAX_LEVELS 1500
#define MAX_NUM 10000
double* averageOfLevels(struct TreeNode* root, int* returnSize){
    double* tempResult = (double*)malloc(MAX_LEVELS * sizeof(double));
    struct TreeNode** tempQueue = (struct TreeNode**)malloc(MAX_NUM * sizeof(struct TreeNode*));
    int in = 0;
    int out = 0;
    tempQueue[0] = root;
    in++;
    int flag  = in;
    double sumOfLevel = 0;
    int cntOfLevel = 0;
    int cnt = 0;
    while (out < in) {
        sumOfLevel += tempQueue[out]->val;
        cntOfLevel++;
        if (tempQueue[out]->left != NULL)
            tempQueue[in++] = tempQueue[out]->left;
        if (tempQueue[out]->right != NULL)
            tempQueue[in++] = tempQueue[out]->right;
        out++;
        if (flag == out) {
            tempResult[cnt++] = sumOfLevel/(double)cntOfLevel;
            sumOfLevel = 0;
            cntOfLevel = 0;
            flag = in;
        }
    }
    double* result = (double*)malloc(cnt * sizeof(double));
    memcpy(result, tempResult, cnt * sizeof(double));
    *returnSize = cnt;
    free(tempResult);
    free(tempQueue);
    return result;
}

//643
double findMaxAverage(int* nums, int numsSize, int k) {
    double m;
    long sum = 0,temp;
    for(int i = 0;i < k;i++)
        sum = sum + nums[i];
    temp = sum;
    for(int i = k;i < numsSize;i++){
        temp = temp - nums[i-k] + nums[i];
        if(sum < temp)
            sum = temp;
    }
    m = (double) sum / k;
    return m;
}

//645
int* findErrorNums(int* nums, int numsSize, int* returnSize) {
    int *result=(int*)malloc(2*sizeof(int));
    int *count=calloc(numsSize+1,sizeof(int));
    int i,rep,sum=0,lost;
    for(i=0;i<numsSize;i++){
        count[nums[i]]++;
        if(2==count[nums[i]])
            rep=nums[i];
        sum=sum+nums[i];
    }
    lost=numsSize*(numsSize+1)/2-(sum-rep);
    result[0]=rep;
    result[1]=lost;
    *returnSize=2;
    return result;
}

//647
int countSubstrings(char * s){
    int len = strlen(s);
    int* dp = (int*)malloc(sizeof(int)*len);
    int cnt= 0;
    for(int i = 0; i < len; i++){
        dp[i] = 1;
        cnt++;
        for(int j = 0; j < i; j++){
            if(s[j] == s[i] && dp[j+1] == 1){
                dp[j]= 1;
                cnt++;
            }else{
                dp[j] = 0;
            }
        }
    }    
    return cnt;
}

//653
int gk = 0;
struct TreeNode * groot = NULL;
bool gret = false;
void find (struct TreeNode * root, int n){
    if (root == NULL){return ;}
    if (root->val == n){
        gret = true;
        return;
    }
    find((n > root->val)?root->right:root->left, n);
}
void dfs(struct TreeNode * root){
    if (root == NULL || gret){return;}
    int diff = gk - root->val;
    if (diff != root->val)
        find(groot, diff);
    dfs(root->left);
    dfs(root->right);
}
bool findTarget(struct TreeNode* root, int k){
    groot = root;
    gk = k;
    gret = false;
    dfs(root);
    return gret;
}

//657
bool judgeCircle(char * moves) {
    char c;
    int x = 0, y = 0;
    while ((c = *moves++) != '\0') {
        switch(c) {
            case 'L': --x; break;
            case 'R': ++x; break;
            case 'D': --y; break;
            case 'U': ++y; break;
        }
    }
    return x == 0 && y == 0;
}

//661
int diff[8][2] = {{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1}};
int getR(int** M, int x, int y, int X, int Y) {
    int i;
    int sum = M[x][y];
    int c = 1;
    for(i = 0; i < 8; i++)
        if(x + diff[i][0] >= 0 && x + diff[i][0] < X && \
           y + diff[i][1] >= 0 && y + diff[i][1] < Y) { 
            sum += M[x + diff[i][0]][y + diff[i][1]];
            c++;
        }
    return sum /= c;
}
 
int** imageSmoother(int** M, int MSize, int* MColSize, int* returnSize, int** returnColumnSizes){
    int i, j;
    int **R = malloc(sizeof(int *) * MSize);
    *returnColumnSizes =  malloc(sizeof(int) * MSize);
    for(i = 0; i < MSize; i++) {
        R[i] = malloc(sizeof(int) * *MColSize);
        (*returnColumnSizes)[i] = *MColSize;
    }
    for(i = 0;i < MSize; i++) {
        for(j = 0; j < *MColSize; j++)
            R[i][j] = getR(M, i, j, MSize, *MColSize);
    }
    *returnSize = MSize;
    return R;
}

//665
bool checkPossibility(int* nums, int numsSize){
    if(numsSize <= 2) return true;
    int res = 0;
    if(nums[0] > nums[1]){
        res++;
        nums[0] = nums[1];
    }
    for(int i=1;i<numsSize-1;i++){
        if(nums[i] > nums[i+1]){
            res++;
            if(res > 1) return false;
            else if(nums[i-1] > nums[i+1]) nums[i+1] = nums[i];
            else nums[i] = nums[i-1];
        }
    }
    return true;
}

//669
struct TreeNode* trimBST(struct TreeNode* root, int L, int R){
    if(root==NULL) return root;
    if(L>root->val) return trimBST(root->right,L,R);
    if(R<root->val) return trimBST(root->left,L,R);
    root->left=trimBST(root->left,L,R);
    root->right=trimBST(root->right,L,R);
    return root;
}

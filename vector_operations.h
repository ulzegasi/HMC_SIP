using namespace std;

//// Sum of elements in a vector
template<typename elemType> elemType vreduce(const vector<elemType> &vec)
{
	elemType sum_of_elems = 0;
	for_each(vec.begin(), vec.end(), [&](elemType elem){sum_of_elems += elem;});
	return sum_of_elems;
}

//// Sums two vectors element-wise
template<typename elemType> vector<elemType> vsum(const vector<elemType> &vec1, const vector<elemType> &vec2)
{
	vector<elemType> vec_plus_vec(vec1.size());
	transform(vec1.begin(), vec1.end(), vec2.begin(), vec_plus_vec.begin(), plus<elemType>());
	return vec_plus_vec;
}

//// Square root of elements of a vector
template<typename elemType> vector<double> vsqrt(const vector<elemType> &vec)
{
	vector<double> sqrt_vec(vec.size());
	vector<double>::iterator iter = sqrt_vec.begin();
	for_each(vec.begin(), vec.end(), [&](elemType elem){
		*iter = sqrt(elem); ++iter;});
	return sqrt_vec;
}

//// Square of elements of a vector
template<typename elemType> vector<elemType> vsquare(const vector<elemType> &vec)
{
	vector<elemType> vec2(vec.size());
	auto iter = vec2.begin();
	for_each(vec.begin(), vec.end(), [&](elemType elem){
		*iter = pow(elem,2.0); ++iter;});
	return vec2;
}

//// Multiplies a vector by a constant
template<typename elemType> vector<elemType> ctimes(elemType val, const vector<elemType> &vec)
{
	vector<elemType> vec_times_val(vec.size());
	auto iter = vec_times_val.begin();
	for_each(vec.begin(), vec.end(), [&](elemType elem){
		*iter = val*elem; ++iter;});
	return vec_times_val;
}

//// Multiplies two vectors element-wise
template<typename elemType> vector<elemType> vtimes(const vector<elemType> &vec1, const vector<elemType> &vec2)
{
	vector<elemType> vec_times_vec(vec1.size());
	transform(vec1.begin(), vec1.end(), vec2.begin(), vec_times_vec.begin(), multiplies<elemType>());
	return vec_times_vec;
}

//// Divides a vector by a constant
template<typename elemType> vector<elemType> cdiv(elemType val, const vector<elemType> &vec)
{
	vector<elemType> vec_div_val(vec.size());
	auto iter = vec_div_val.begin();
	for_each(vec.begin(), vec.end(), [&](elemType elem){
		*iter = elem/val; ++iter;});
	return vec_div_val;
}

//// Divides two vectors element-wise
template<typename elemType> vector<elemType> vdiv(const vector<elemType> &vec1, const vector<elemType> &vec2)
{
	vector<elemType> vec_div_vec(vec1.size());
	transform(vec1.begin(), vec1.end(), vec2.begin(), vec_div_vec.begin(), divides<elemType>());
	return vec_div_vec;
}
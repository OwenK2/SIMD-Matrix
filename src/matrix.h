#include <immintrin.h>
#include <type_traits>
#include <initializer_list>
#include <iostream>
#include <cstring>

#define type_equal(type2) std::is_same<T, type2>::value


template<typename T, size_t R, size_t C>
class Matrix {
public:
	static constexpr size_t size = R * C;
	static constexpr size_t rows = R;
	static constexpr size_t cols = C;

	Matrix() {}
	Matrix(T init) {
		for(size_t i = 0;i < size;++i) {
			m_data[i] = init;
		}
	}	
	Matrix(std::initializer_list<T> list) {
		if(list.size() == size) {
			size_t i = 0;
			for(auto l : list) {
				m_data[i++] = l;
			}
		}
		else {
			fprintf(stderr, "\e[31m[ERROR] Initializer list of size %zu does not match matrix of size %zux%zu\e[0m\n", list.size(), rows, cols);
		}
	}
	Matrix(const Matrix<T, R, C>& rhs) {memcpy(m_data, rhs.m_data, sizeof(m_data));}
	Matrix(const Matrix<T, R, C>&& rhs) {memcpy(m_data, rhs.m_data, sizeof(m_data));}
	~Matrix() {}


	T& at(size_t n) const {return m_data[n];}
	T& first(size_t n) const {return m_data[0];}
	T& last(size_t n) const {return m_data[size-1];}
	T& operator[](size_t n) {return m_data[n];}
	Matrix& operator=(const Matrix<T,R,C>& rhs) {memcpy(m_data, rhs.m_data, sizeof(m_data));return *this;}
	Matrix& operator=(const Matrix<T,R,C>&& rhs) {memcpy(m_data, rhs.m_data, sizeof(m_data));return *this;}


	bool operator==(const Matrix<T,R,C>& rhs) const {
		for(size_t i = 0;i < size;++i) {
			if(m_data[i] != rhs.m_data[i])
				return false;
		}
		return true;
	}
	bool operator!=(const Matrix<T,R,C>& rhs) const {return !(*this == rhs);}

	Matrix<T,R,C> operator+(const Matrix<T,R,C>& rhs) {
		Matrix<T,R,C> result;
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				v2 = _mm256_load_si256((__m256i const*)&rhs.m_data[i]);
				if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v1 = _mm256_add_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_add_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_add_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_add_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2;
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v2 = _mm256_load_ps(&rhs.m_data[i]);
				v1 = _mm256_add_ps(v1, v2);
				_mm256_store_ps(&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2;
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v2 = _mm256_load_pd(&rhs.m_data[i]);
				v1 = _mm256_add_pd(v1, v2);
				_mm256_store_pd(&result.m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			result.m_data[i] = m_data[i] + rhs.m_data[i];
		}
		return result;
	}

	Matrix<T,R,C>& operator+=(const Matrix<T,R,C>& rhs) {
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				v2 = _mm256_load_si256((__m256i const*)&rhs.m_data[i]);
				if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v1 = _mm256_add_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_add_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_add_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_add_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2;
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v2 = _mm256_load_ps(&rhs.m_data[i]);
				v1 = _mm256_add_ps(v1, v2);
				_mm256_store_ps(&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2;
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v2 = _mm256_load_pd(&rhs.m_data[i]);
				v1 = _mm256_add_pd(v1, v2);
				_mm256_store_pd(&m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			m_data[i] += rhs.m_data[i];
		}
		return *this;
	}

	Matrix<T,R,C> operator-(const Matrix<T,R,C>& rhs) {
		Matrix<T,R,C> result;
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				v2 = _mm256_load_si256((__m256i const*)&rhs.m_data[i]);
				if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v1 = _mm256_sub_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_sub_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_sub_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_sub_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2;
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v2 = _mm256_load_ps(&rhs.m_data[i]);
				v1 = _mm256_sub_ps(v1, v2);
				_mm256_store_ps(&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2;
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v2 = _mm256_load_pd(&rhs.m_data[i]);
				v1 = _mm256_sub_pd(v1, v2);
				_mm256_store_pd(&result.m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			result.m_data[i] = m_data[i] - rhs.m_data[i];
		}
		return result;
	}

	Matrix<T,R,C>& operator-=(const Matrix<T,R,C>& rhs) {
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				v2 = _mm256_load_si256((__m256i const*)&rhs.m_data[i]);
				if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v1 = _mm256_sub_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_sub_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_sub_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_sub_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2;
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v2 = _mm256_load_ps(&rhs.m_data[i]);
				v1 = _mm256_sub_ps(v1, v2);
				_mm256_store_ps(&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2;
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v2 = _mm256_load_pd(&rhs.m_data[i]);
				v1 = _mm256_sub_pd(v1, v2);
				_mm256_store_pd(&m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			m_data[i] -= rhs.m_data[i];
		}
		return *this;
	}

	Matrix<T,R,C> operator+(T scalar) {
		Matrix<T,R,C> result;
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v2 = _mm256_set1_epi64x(scalar);}
			else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v2 = _mm256_set1_epi32(scalar);}
			else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v2 = _mm256_set1_epi16(scalar);}
			else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v2 = _mm256_set1_epi8(scalar);}
			
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v1 = _mm256_add_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_add_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_add_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_add_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2 = _mm256_broadcast_ss(&scalar);
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v1 = _mm256_add_ps(v1, v2);
				_mm256_store_ps(&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2 = _mm256_broadcast_sd(&scalar);
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v1 = _mm256_add_pd(v1, v2);
				_mm256_store_pd(&result.m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			result.m_data[i] = m_data[i] + scalar;
		}
		return result;
	}

	Matrix<T,R,C>& operator+=(T scalar) {
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v2 = _mm256_set1_epi64x(scalar);}
			else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v2 = _mm256_set1_epi32(scalar);}
			else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v2 = _mm256_set1_epi16(scalar);}
			else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v2 = _mm256_set1_epi8(scalar);}
			
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v1 = _mm256_add_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_add_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_add_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_add_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2 = _mm256_broadcast_ss(&scalar);
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v1 = _mm256_add_ps(v1, v2);
				_mm256_store_ps(&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2 = _mm256_broadcast_sd(&scalar);
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v1 = _mm256_add_pd(v1, v2);
				_mm256_store_pd(&m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			m_data[i] += scalar;
		}
		return *this;
	}

	Matrix<T,R,C> operator-(T scalar) {
		Matrix<T,R,C> result;
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v2 = _mm256_set1_epi64x(scalar);}
			else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v2 = _mm256_set1_epi32(scalar);}
			else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v2 = _mm256_set1_epi16(scalar);}
			else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v2 = _mm256_set1_epi8(scalar);}
			
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v1 = _mm256_sub_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_sub_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_sub_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_sub_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2 = _mm256_broadcast_ss(&scalar);
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v1 = _mm256_sub_ps(v1, v2);
				_mm256_store_ps(&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2 = _mm256_broadcast_sd(&scalar);
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v1 = _mm256_sub_pd(v1, v2);
				_mm256_store_pd(&result.m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			result.m_data[i] = m_data[i] - scalar;
		}
		return result;
	}

	Matrix<T,R,C>& operator-=(T scalar) {
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v2 = _mm256_set1_epi64x(scalar);}
			else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v2 = _mm256_set1_epi32(scalar);}
			else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v2 = _mm256_set1_epi16(scalar);}
			else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v2 = _mm256_set1_epi8(scalar);}
			
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v1 = _mm256_sub_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_sub_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_sub_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_sub_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2 = _mm256_broadcast_ss(&scalar);
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v1 = _mm256_sub_ps(v1, v2);
				_mm256_store_ps(&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2 = _mm256_broadcast_sd(&scalar);
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v1 = _mm256_sub_pd(v1, v2);
				_mm256_store_pd(&m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			m_data[i] -= scalar;
		}
		return *this;
	}

	Matrix<T,R,C> operator*(T scalar) {
		Matrix<T,R,C> result;
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v2 = _mm256_set1_epi64x(scalar);}
			else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v2 = _mm256_set1_epi32(scalar);}
			else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v2 = _mm256_set1_epi16(scalar);}
			else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v2 = _mm256_set1_epi8(scalar);}
			
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				if constexpr(type_equal(int64_t) || type_equal(uint64_t)) {v1 = _mm256_mullo_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_mullo_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_mullo_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_mullo_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2 = _mm256_broadcast_ss(&scalar);
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v1 = _mm256_mul_ps(v1, v2);
				_mm256_store_ps(&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2 = _mm256_broadcast_sd(&scalar);
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v1 = _mm256_mul_pd(v1, v2);
				_mm256_store_pd(&result.m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			result.m_data[i] = m_data[i] * scalar;
		}
		return result;
	}
	Matrix<T,R,C>& operator*=(T scalar) {
		size_t i = 0;
		if constexpr(type_equal(uint64_t) || type_equal(int64_t) || type_equal(uint32_t) || type_equal(int32_t) || type_equal(uint16_t) || type_equal(int16_t) || type_equal(uint8_t) || type_equal(int8_t)) {
			__m256i v1;
			__m256i v2;
			if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {v2 = _mm256_set1_epi64x(scalar);}
			else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v2 = _mm256_set1_epi32(scalar);}
			else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v2 = _mm256_set1_epi16(scalar);}
			else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v2 = _mm256_set1_epi8(scalar);}
			
			for(;i+256 / (sizeof(T) * 8) < size;i += 256 / (sizeof(T) * 8)) {
				v1 = _mm256_load_si256((__m256i const*)&m_data[i]);
				if constexpr(type_equal(int64_t) || type_equal(uint64_t)) {v1 = _mm256_mullo_epi64(v1, v2);}
				else if constexpr(type_equal(uint32_t) || type_equal(int32_t)) {v1 = _mm256_mullo_epi32(v1, v2);}
				else if constexpr(type_equal(uint16_t) || type_equal(int16_t)) {v1 = _mm256_mullo_epi16(v1, v2);}
				else if constexpr(type_equal(uint8_t) || type_equal(int8_t)) {v1 = _mm256_mullo_epi8(v1, v2);}
				_mm256_store_si256((__m256i*)&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2 = _mm256_broadcast_ss(&scalar);
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v1 = _mm256_mul_ps(v1, v2);
				_mm256_store_ps(&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2 = _mm256_broadcast_sd(&scalar);
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v1 = _mm256_mul_pd(v1, v2);
				_mm256_store_pd(&m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			m_data[i] *= scalar;
		}
		return *this;
	}

	Matrix<T,R,C> operator/(T scalar) {
		Matrix<T,R,C> result;
		size_t i = 0;
		if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2 = _mm256_broadcast_ss(&scalar);
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v1 = _mm256_div_ps(v1, v2);
				_mm256_store_ps(&result.m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2 = _mm256_broadcast_sd(&scalar);
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v1 = _mm256_div_pd(v1, v2);
				_mm256_store_pd(&result.m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			result.m_data[i] = m_data[i] / scalar;
		}
		return result;
	}

	Matrix<T,R,C>& operator/=(T scalar) {
		size_t i = 0;
		if constexpr(type_equal(float) && sizeof(float) == 4) {
			__m256 v1;
			__m256 v2 = _mm256_broadcast_ss(&scalar);
			for(;i+8 < size;i += 8) {
				v1 = _mm256_load_ps(&m_data[i]);
				v1 = _mm256_div_ps(v1, v2);
				_mm256_store_ps(&m_data[i], v1);
			}
		}
		else if constexpr(type_equal(double) && sizeof(double) == 8) {
			__m256d v1;
			__m256d v2 = _mm256_broadcast_sd(&scalar);
			for(;i+4 < size;i += 4) {
				v1 = _mm256_load_pd(&m_data[i]);
				v1 = _mm256_div_pd(v1, v2);
				_mm256_store_pd(&m_data[i], v1);
			}
		}
		for(;i < size;++i) {
			m_data[i] /= scalar;
		}
		return *this;
	}

	template<size_t P>
	Matrix<T,R,P> operator*(Matrix<T,C,P>& rhs) {
		Matrix<T,R,P> result;
		if constexpr(type_equal(float)) {

		}
		else {
			size_t i, j, k;
	    for(i = 0; i < R; ++i) {
	      for(j = 0; j < P; ++j) {
	      	result.m_data[i*P + j] = 0;
	        for(k = 0; k < C; ++k) {
	          result.m_data[i*P + j] += m_data[i*C + k] * rhs.m_data[k*P + j];
	        }
	      }
	    }
	  }
	  return result;
	}

	Matrix<T,R,C>& abs() {
		if constexpr(std::is_signed<T>::value) {
			for(size_t i = 0;i < size;++i) {
				if constexpr(type_equal(int64_t) || type_equal(int32_t) || type_equal(int16_t) || type_equal(int8_t)) {
			    T s = m_data[i] >> (sizeof(T) * 8 - 1);
			    m_data[i] = (m_data[i] ^ s) - s;
				}
				else {
					if(m_data[i] < 0) 
						m_data[i] = -m_data[i];
				}
			}
		}
		return *this;
	}




	// Iterator
	class iterator {
	private:
		T* m_ptr;
	public:
		iterator(T* ptr) : m_ptr(ptr) {}
		iterator operator++() {iterator i = *this;++m_ptr; return i;}
		iterator& operator++(int) {++m_ptr; return *this;}
		iterator operator--() {iterator i = *this;--m_ptr; return i;}
		iterator& operator--(int) {--m_ptr; return *this;}
		T& operator*() {return *m_ptr;}
		inline bool operator==(const iterator& rhs) const {return m_ptr == rhs.m_ptr;}
		inline bool operator !=(const iterator& rhs) const {return m_ptr != rhs.m_ptr;}
	};

	iterator begin() {
		return iterator(m_data);
	}
	iterator end() {
		return iterator(m_data+size);
	}

	void print(uint8_t padding = 1, uint8_t precision = 2) const {
		uint16_t totalWidth = padding;
		uint8_t colsWidth[cols];
		if constexpr(!std::is_floating_point<T>::value) {precision = 0;}

		uint8_t width = 0;
		for(size_t i = 0;i < cols;++i) {
			colsWidth[i] = 0;
			for(size_t j = 0;j < rows;++j) {
				width = getWidth(m_data[i+j*cols], precision) + padding;
				if(width > colsWidth[i]) {colsWidth[i] = width;}
			}	
			totalWidth += colsWidth[i];
		}

		printf("┌%*c┐\n", totalWidth, ' ');

		for(size_t i = 0;i < rows;++i) {
			printf("%-*s", padding + 3, "│");
			for(size_t j = 0;j < cols;++j) {
				uint8_t pad = colsWidth[j];
				if constexpr(std::is_floating_point<T>::value) {
					printf("%-*.*f", pad, precision, m_data[i * cols + j]);
				}
				else if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {
					printf("%-*lld", pad, m_data[i * cols + j]);
				}
				else {
					printf("%-*d", pad, m_data[i * cols + j]);
				}
			}
			printf("│\n");
		}

		printf("└%*c┘\n", totalWidth, ' ');

	}


private:
	T m_data[size] __attribute__((aligned(32)));


	inline uint8_t getWidth(T x, uint8_t precision) const {  
		if constexpr(std::is_floating_point<T>::value) {
			return snprintf(NULL, 0, "%.*f", precision, x);
		}
		else if constexpr(type_equal(uint64_t) || type_equal(int64_t)) {
			return snprintf(NULL, 0, "%lld", x);
		}
		else {
			return snprintf(NULL, 0, "%d", x);
		} 
	}

	static inline __m256i _mm256_mullo_epi8(__m256i a, __m256i b) {
    __m256i dst_even = _mm256_mullo_epi16(a, b);
    __m256i dst_odd = _mm256_mullo_epi16(_mm256_srli_epi16(a, 8),_mm256_srli_epi16(b, 8));
		return _mm256_or_si256(_mm256_slli_epi16(dst_odd, 8), _mm256_and_si256(dst_even, _mm256_set1_epi16(0xFF)));
	}

	static inline __m256i _mm256_mullo_epi64(__m256i a, __m256i b) {
    __m256i ac = _mm256_mul_epu32(a, b);
    __m256i alo = _mm256_srli_epi64(a, 32);
    __m256i bc = _mm256_mul_epu32(alo, b);
    __m256i d = _mm256_srli_epi64(b, 32);
    __m256i ad = _mm256_mul_epu32(a, d);
    __m256i high = _mm256_add_epi64(bc, ad);
    high = _mm256_slli_epi64(high, 32);
    return _mm256_add_epi64(high, ac);
	}

	template<typename U, size_t X, size_t Y>
  friend class Matrix;
};
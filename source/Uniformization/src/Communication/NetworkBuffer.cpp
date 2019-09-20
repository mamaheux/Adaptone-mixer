#include <Uniformization/Communication/NetworkBuffer.h>

using namespace adaptone;
using namespace std;

NetworkBufferView::NetworkBufferView(uint8_t* data, size_t size) : m_data(data), m_size(size)
{

}

NetworkBufferView::~NetworkBufferView()
{
}

NetworkBuffer::NetworkBuffer(size_t size) : m_size(size)
{
    m_data = new uint8_t[size];
}

NetworkBuffer::~NetworkBuffer()
{
    delete[] m_data;
}
